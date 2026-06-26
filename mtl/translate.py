"""Pluggable Japanese->English translators. All run on-device.

Add a new backend by writing a Translator subclass and registering it in
TRANSLATORS. The pipeline only depends on the .translate(str) -> str contract,
so swapping engines (opus-mt -> Sugoi -> a local LLM) is a one-line change.
"""

from __future__ import annotations

import functools
import json
import os
import re
import urllib.request
from typing import Dict, List, Type

from .device import get_device


class Translator:
    def translate(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a page's worth of bubbles. Default loops per-bubble;
        backends that benefit from context/throughput (LLMs) override this."""
        return [self.translate(t) for t in texts]


class NullTranslator(Translator):
    """Passes the Japanese through unchanged. Useful for testing detection,
    clustering, and rendering without loading a translation model."""

    def translate(self, text: str) -> str:
        return text


class OpusMTTranslator(Translator):
    """Helsinki-NLP/opus-mt-ja-en (Marian). Small, fully offline after the
    one-time model download. The default; weaker on slang but a solid baseline.
    """

    MODEL = "Helsinki-NLP/opus-mt-ja-en"

    def __init__(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(self.MODEL)
        self.model = MarianMTModel.from_pretrained(self.MODEL)
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str) -> str:
        import torch

        if not text.strip():
            return ""
        batch = self.tokenizer([text], return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            # Beam search + anti-repetition is essential: greedy decoding makes
            # these Marian models (FuguMT especially) degenerate into repetition
            # loops, particularly on the noisy text manga OCR produces.
            generated = self.model.generate(
                **batch,
                max_length=512,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


class FuguMTTranslator(OpusMTTranslator):
    """FuguMT (staka/fugumt-ja-en). A Marian model trained specifically for
    Japanese->English; markedly better register and far less hallucination than
    opus-mt, and the practical self-hosted stand-in for Sugoi (which is
    fairseq-based and awkward to host). Same transformers API as opus-mt, so it
    just swaps the model name. Fully offline after the one-time download."""

    MODEL = "staka/fugumt-ja-en"


class OllamaTranslator(Translator):
    """Local LLM translation via Ollama. Fully self-hosted (talks to the local
    Ollama daemon at localhost:11434) and far higher quality than the NMT models
    — natural, casual, manga-appropriate English. Ollama uses the machine's GPU
    (Metal on Mac, ROCm/AMD where available), so it's also fast once warm.

    A whole page's bubbles go in one request so the model sees context and we
    pay model overhead once per page rather than per bubble.

    Model is configurable via --ollama-model / MTL_OLLAMA_MODEL. Default
    qwen3:1.7b (validated fast+good on the AMD APU); use qwen3:8b or larger on a
    Mac for better quality.
    """

    HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    SYSTEM = (
        "You are a professional Japanese-to-English manga translator. You will be "
        "given a numbered list of Japanese lines from one manga page. Translate each "
        "into natural, casual English as it would appear in an official English "
        "release. The OCR may be imperfect — infer the most sensible reading. Output "
        "ONLY a JSON array of strings, one translation per input line, in the same "
        "order and with the same count. No notes, no markdown, no numbering."
    )

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("MTL_OLLAMA_MODEL", "qwen3:1.7b")
        # Fail fast with a clear message if the daemon isn't up.
        try:
            urllib.request.urlopen(f"{self.HOST}/api/tags", timeout=5).read()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot reach Ollama at {self.HOST} — is `ollama serve` running and "
                f"is '{self.model}' pulled (`ollama pull {self.model}`)? ({exc})"
            ) from exc

    def _chat(self, user: str, timeout: int = 600) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "think": False,  # qwen3 is a thinking model; disable for translation
                "stream": False,
                "options": {"temperature": 0.3},
                "messages": [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.HOST}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = json.load(resp)["message"]["content"]
        # Strip any stray <think> block defensively.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()

    @staticmethod
    def _parse(raw: str, n: int) -> List[str]:
        s = raw.strip()
        if "[" in s and "]" in s:
            try:
                arr = json.loads(s[s.find("[") : s.rfind("]") + 1])
                if isinstance(arr, list):
                    out = [str(x).strip() for x in arr]
                    return (out + [""] * n)[:n]
            except Exception:  # noqa: BLE001
                pass
        # Fallback: one translation per non-empty line, strip any numbering.
        lines = [
            re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
            for ln in raw.splitlines()
            if ln.strip()
        ]
        return (lines + [""] * n)[:n]

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return self._parse(self._chat(numbered), len(texts))


TRANSLATORS: Dict[str, Type[Translator]] = {
    "qwen": OllamaTranslator,
    "fugumt": FuguMTTranslator,
    "opus": OpusMTTranslator,
    "null": NullTranslator,
}


@functools.lru_cache(maxsize=None)
def get_translator(name: str) -> Translator:
    if name not in TRANSLATORS:
        raise ValueError(
            f"Unknown translator '{name}'. Options: {', '.join(TRANSLATORS)}"
        )
    return TRANSLATORS[name]()
