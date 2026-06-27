# CLAUDE.md — manga-translator

Working notes for resuming this project on another machine (next stop: a Mac).

## What this is

A **completely self-hosted** utility that translates Japanese manga pages to
English and draws the English **in place** over the original art. No cloud APIs,
ever — models download once from public repos, then everything runs offline.
Input photos live in `input/`, results are written to `output/`.

## Hard constraints (these came from the user — honor them)

1. **Completely self-hosted. No cloud translation APIs of any kind.** This is
   non-negotiable and was stated emphatically.
2. **No inpainting.** Do NOT try to erase the Japanese. Draw English over the
   art; surrounding Japanese showing through is *wanted* ("nice for atmosphere").
3. **No speech-bubble box (current default).** The English text carries its own
   white halo (stroke) for legibility, so the opaque/translucent box was removed.
   `--bubble-opacity` still exists (0.0 = no box, default; up to 1.0 = solid) in
   case we want it back, but default is 0.
4. **Translation quality is the top priority and is currently the weak point.**
   FuguMT (and opus-mt) are too poor. We are moving to a **local LLM via Ollama**
   (see Open work). Don't regress to NMT as the default once the LLM path lands.

## Current status

Working end-to-end on the Linux origin machine (CPU for OCR/detect; see Devices):
- detect → cluster → OCR → translate → render all functional.
- ~20–35s/page on CPU (OCR is the slow part; LLM translate is GPU-fast).
- Default render is **no box** — haloed English directly over the art.

**Translation is now solved via a local LLM.** The `qwen` translator
(`OllamaTranslator` in `mtl/translate.py`) is **implemented and the default**. It
batches a whole page into one Ollama `/api/chat` call (`think:false`), parses a
JSON array back, and produces natural, casual English — a night-and-day jump over
FuguMT/opus (which remain registered as `fugumt`/`opus` but are low quality; keep
them only as offline fallbacks). `null` = passthrough for debugging.

The default model is **platform-aware** (`_DEFAULT_OLLAMA_MODEL` in `translate.py`):
**`qwen3:8b` on macOS** (Metal handles it — better quality) and **`qwen3:1.7b` on
Linux**, validated fast+good on the origin AMD APU (GPU via Ollama, confirmed with
radeontop). `qwen3:4b`+ overflow that APU's VRAM and fall back to slow CPU — hence
the smaller Linux default. Override either with `--ollama-model` / `MTL_OLLAMA_MODEL`
(`ollama pull qwen3:8b` once on the Mac first).

**Detection/clustering — not OCR — was the main quality gap on dense pages.**
manga-ocr reads *tight single-text crops* accurately; the garbled output (e.g.
ケータイ "cell phone" rendered as "Kaitain") came from **clustering merging
several distinct texts into one crop**, which manga-ocr (a one-region reader)
can't parse — and the resulting oversized box also tripped the area filter, so
whole regions vanished. Fixed by looser detection (`text_threshold` 0.6→0.4) +
tighter clustering (`gap_ratio` 0.3→0.15); a 4-koma test page went 2 → 7 rendered
translations and the Kaitain misread disappeared. A per-crop head-to-head
confirmed **manga-ocr beats a general vision-LLM (gemma3:12b)** — gemma3
hallucinates on tight crops (でんじろう, even Cyrillic), so do NOT swap it in as
the reader. Residual hard tail: a few densely-packed regions (title chained to
sound FX, stacked handwritten asides) still merge/garble; the next lever is an
OCR-specialised VLM (qwen2.5-vl) as a *fallback* for just those crops. Cleaner
inputs (crop to the read page) still help for free.

## Architecture / pipeline

`run.py` (CLI) → `mtl/pipeline.py:translate_page` runs the stages:

1. **detect** (`mtl/detect.py`) — EasyOCR's CRAFT detector only (not its OCR).
   Returns text-line boxes. Runs on a downscaled copy (`--detect-max-side`,
   default 2000) for speed; boxes are scaled back to full res. `text_threshold`
   0.4 (looser than EasyOCR's 0.6 default) to catch faint/handwritten lettering.
2. **cluster** (`mtl/cluster.py`) — union-find merge of line boxes into bubbles.
   **Pads by the box's SHORTER side** (≈ column width for vertical text), NOT
   height — padding by height chained the whole page into one blob. `gap_ratio`
   default 0.15 (was 0.3; 0.3 over-merged distinct texts on dense 4-koma pages).
3. **ocr** (`mtl/ocr.py`) — manga-ocr reads each whole-bubble crop (best local
   manga reader; handles vertical/multiline).
4. **translate** (`mtl/translate.py`) — pluggable `Translator` subclasses in a
   `TRANSLATORS` registry. Contract is `.translate(str) -> str`.
5. **render** (`mtl/render.py`) — `render_page` composites once via an RGBA
   overlay: opaque black text with a white halo, optional translucent box.

`mtl/device.py` — picks `cuda` if torch sees it, else `cpu`. **Only knows
cuda/cpu** — see Devices for the Mac (MPS) gap.

### File map
- `run.py` — CLI entry. Flags: `-i/-o`, `-t/--translator`, `--detect-max-side`,
  `--bubble-opacity`, `--dump-text`.
- `mtl/` — `device, detect, cluster, ocr, translate, render, pipeline`.
- `setup.sh` — creates an **isolated** `.venv` (no `--system-site-packages`),
  installs `requirements.txt`.
- `debug_detect.py`, `debug_cluster.py` — tuning visualizers; write annotated
  JPGs to `/tmp/mt_smoke_out/`. Safe to delete; handy to keep.

## Setup / run

```bash
./setup.sh
source .venv/bin/activate
python run.py                       # input/ -> output/, default translator
python run.py --dump-text           # also write per-page JA/EN .txt
python run.py -t null               # skip translation (debug detect/cluster/render)
python run.py --bubble-opacity 0.6  # bring back a translucent box
```

## Key decisions & gotchas (hard-won — don't relearn these)

- **Greedy decoding breaks the Marian NMT models.** FuguMT/opus degenerate into
  repetition loops ("swol swol swol…") with greedy. `translate()` uses beam
  search + `no_repeat_ngram_size=3` + `repetition_penalty=1.5`. Keep it.
- **transformers is 5.x here** (5.12.x). `pipeline("translation")` was removed —
  the `"translation"` task no longer exists. Use `MarianMTModel`/tokenizer
  directly (we do).
- **Clustering pads by shorter side, `gap_ratio` 0.15.** Height-based padding
  merged the whole page (fixed first). The original 0.3 still over-merged
  distinct texts (title, SFX, asides) into panel blobs on dense 4-koma pages;
  0.15 separates them while keeping a bubble's own columns together. Pages with
  normal bubble whitespace are unaffected — their bubble counts barely move
  (verified across all 5 sample pages).
- **Oversized-box filter** (`max_box_area_frac=0.10` in `pipeline.py`) drops
  giant merges — these are usually the *facing page* bleeding into the photo.
- **Inputs are angled phone photos** with a thumb and the facing page partly
  visible. This causes OCR slips (e.g., manga-ocr misreads stylized kanji) and
  facing-page bleed. **Cropping each photo to just the read page would improve
  OCR and detection for free** — worth suggesting to the user.
- **Rendering composites once per page** (`render_page`) via RGBA overlay so text
  is fully opaque while any box stays translucent. Don't go back to per-box
  mutation — alpha compositing needs the single-overlay approach.
- Fonts: `render.py` searches common DejaVu/Liberation paths, falls back to
  PIL's bitmap font. **On the Mac add a mac font path** (e.g. Helvetica/Arial) to
  `_FONT_CANDIDATES` or text falls back to the ugly bitmap font.

## Devices (origin Linux vs target Mac)

- **Origin (Linux/AMD):** torch installed is a **CUDA build** (`+cuXXX`) that
  **cannot see the AMD GPU**, so torch (OCR + detection) runs on **CPU**. Ollama,
  however, drives the AMD GPU through its own backend, so the LLM is accelerated.
  No system ROCm (`rocminfo` absent).
- **Target (Mac / Apple Silicon):** Ollama uses **Metal** → fast LLM. For torch
  (OCR/detection), Apple Silicon uses **MPS**, but `device.py` only checks
  `torch.cuda`. **TODO on Mac:** extend `get_device()` to return `"mps"` when
  `torch.backends.mps.is_available()`, and pass the right device to manga-ocr
  (`MangaOcr(force_cpu=...)` only toggles cpu/cuda — may need a small patch) and
  EasyOCR. If MPS is fiddly, CPU OCR is fine; the Mac will still "blow through it".
- **The venv is fully isolated — never add `--system-site-packages` back.** It
  was there originally to reuse a system ROCm torch on the Linux origin, but on
  the Mac there's no system torch to reuse; all it did was leak Homebrew's numpy
  in. That mismatched numpy linked Homebrew's `libomp` while the pip torch wheel
  bundles its own — two LLVM OpenMP runtimes in one process, which first aborts
  with `OMP: Error #15 ... libomp.dylib already initialized` and, if you force
  past it with `KMP_DUPLICATE_LIB_OK=TRUE`, then **segfaults** in torch's
  `_load_from_state_dict` while loading model weights. Isolation makes numpy
  come from a pip wheel like everything else (single libomp), so no env-var
  hacks are needed. `setup.sh` now creates the venv with plain `python -m venv`.

## Open work (prioritized)

1. **Residual hard tail: densely-packed regions that clustering still merges**
   (translation solved; detection/clustering tuned — see Current status). On a
   dense 4-koma page a few crops still bundle multiple texts (title+SFX, stacked
   handwritten asides) and OCR garbles them. Options, in order:
   - Cheapest: crop inputs to the read page (de-skew, drop facing page + thumb).
   - Targeted: an OCR-specialised VLM (qwen2.5-vl via Ollama) as a **fallback**
     for *only* the crops manga-ocr garbles (oversized merges / low-confidence).
     Keep manga-ocr as primary — it beat gemma3 head-to-head on tight crops.
   - Smarter clustering: split a too-tall merged box back into sub-regions rather
     than feeding the whole thing to a one-region reader.
2. **Mac Ollama model**: `qwen3:8b` is now the default on macOS (platform-aware,
   see Current status) — just `ollama pull qwen3:8b` once; bump higher with
   `--ollama-model` if you want.
3. Add **MPS** device support for torch on the Mac (see Devices) — optional
   speedup for OCR/detection.
4. Add a **Mac font path** to `render.py:_FONT_CANDIDATES`.
5. Optional: literal **Sugoi** weights via a CTranslate2 backend (user originally
   asked for "Sugoi"; the LLM path superseded it, but offer if they still want it).

## Git

Branch `wip` off `main`. Single initial commit so far; nothing pushed. Commit/push
only when the user asks.
