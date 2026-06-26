"""Pluggable Japanese->English translators. All run on-device.

Add a new backend by writing a Translator subclass and registering it in
TRANSLATORS. The pipeline only depends on the .translate(str) -> str contract,
so swapping engines (opus-mt -> Sugoi -> a local LLM) is a one-line change.
"""

from __future__ import annotations

import functools
from typing import Dict, Type

from .device import get_device


class Translator:
    def translate(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


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
            generated = self.model.generate(**batch, max_length=512)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


TRANSLATORS: Dict[str, Type[Translator]] = {
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
