# manga-translator

A **completely self-hosted** utility that translates Japanese manga pages to
English, drawing the English in place over the original art. No cloud APIs —
models download once from public repos, then everything runs offline with no
per-page network calls.

## Pipeline

1. **Detect** text regions — EasyOCR's CRAFT detector
2. **Cluster** line boxes into speech bubbles
3. **OCR** the Japanese — [manga-ocr](https://github.com/kha-white/manga-ocr)
4. **Translate** JA→EN — pluggable (default: opus-mt; see *Translators*)
5. **Render** English into an opaque rounded box over the bubble (no inpainting;
   surrounding Japanese art is left intact)

## Setup

```bash
./setup.sh
```

This creates `.venv` with `--system-site-packages` so an existing **ROCm/CUDA
torch** is reused rather than re-downloaded. If you have no system torch, a CPU
build is installed automatically.

### AMD GPU note

torch exposes ROCm through the `torch.cuda` API, so an AMD GPU is auto-detected
and used. If your **integrated** GPU isn't enumerated, export an override before
running (the value depends on your gfx arch; `gfx90c`/`gfx1030` are common for
APUs):

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

If the GPU still isn't seen, the pipeline falls back to CPU — fine for batch
work, just slower. The run prints which device it picked.

## Usage

```bash
source .venv/bin/activate
python run.py                      # input/ -> output/
python run.py --translator null    # skip translation (inspect detection only)
python run.py --dump-text          # also write per-page JA/EN .txt files
python run.py -i pages/ -o out/    # custom folders
```

## Translators

Stage 4 is swappable — implement a `Translator` subclass in `mtl/translate.py`
and register it in `TRANSLATORS`. The pipeline only needs
`.translate(str) -> str`.

| name   | engine                         | notes                                   |
|--------|--------------------------------|-----------------------------------------|
| `opus` | Helsinki-NLP/opus-mt-ja-en     | default; small, offline, literal        |
| `null` | passthrough (keeps Japanese)   | for debugging detection/rendering       |

Planned: Sugoi/JParaCrawl (better manga register) and a local-LLM backend.
