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

Benchmark on the origin AMD APU: **`qwen3:1.7b` is the validated default** — fast
(GPU via Ollama, confirmed with radeontop) and good quality. `qwen3:4b` timed out
(too big for the APU's VRAM, fell back to slow CPU). **On the Mac (Metal), bump to
`qwen3:8b` or larger for better quality**: `python run.py --ollama-model qwen3:8b`.

**OCR is now the main quality bottleneck**, not translation. The garbled bits in
output (e.g. a name rendered as nonsense) are manga-ocr misreads of stylized text
on angled phone photos — the LLM faithfully translates whatever OCR hands it.
Biggest available win: cleaner inputs (crop to the read page) and/or a stronger
OCR/VLM step.

## Architecture / pipeline

`run.py` (CLI) → `mtl/pipeline.py:translate_page` runs the stages:

1. **detect** (`mtl/detect.py`) — EasyOCR's CRAFT detector only (not its OCR).
   Returns text-line boxes. Runs on a downscaled copy (`--detect-max-side`,
   default 2000) for speed; boxes are scaled back to full res.
2. **cluster** (`mtl/cluster.py`) — union-find merge of line boxes into bubbles.
   **Pads by the box's SHORTER side** (≈ column width for vertical text), NOT
   height — padding by height chained the whole page into one blob. `gap_ratio`
   default 0.3.
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
- `setup.sh` — creates `.venv --system-site-packages` (to reuse a system
  torch), installs `requirements.txt`.
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
- **Clustering pads by shorter side, gap_ratio 0.3.** Height-based padding merged
  the whole page. This was the single biggest detection-side bug.
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

## Open work (prioritized)

1. **OCR quality is now the top bottleneck** (translation is solved). Options:
   - Cheapest: crop inputs to the read page (de-skew, drop facing page + thumb).
   - Stronger: try a better OCR/VLM step. A local vision-LLM via Ollama (e.g. a
     qwen VL model) could read the bubble crop directly and translate in one
     shot, skipping the manga-ocr → text → LLM error chain entirely. Worth a
     spike on the Mac where the GPU makes VLMs practical.
2. **On the Mac, raise the Ollama model**: `--ollama-model qwen3:8b` (or larger).
   `ollama pull qwen3:8b` first if needed.
3. Add **MPS** device support for torch on the Mac (see Devices) — optional
   speedup for OCR/detection.
4. Add a **Mac font path** to `render.py:_FONT_CANDIDATES`.
5. Optional: literal **Sugoi** weights via a CTranslate2 backend (user originally
   asked for "Sugoi"; the LLM path superseded it, but offer if they still want it).

## Git

Branch `wip` off `main`. Single initial commit so far; nothing pushed. Commit/push
only when the user asks.
