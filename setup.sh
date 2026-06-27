#!/usr/bin/env bash
# One-time setup. Creates a fully ISOLATED venv (no --system-site-packages) so
# the pipeline never depends on whatever Python packages happen to be installed
# system-wide. Mixing a system numpy (which links the system libomp) with the
# pip torch wheel (which bundles its own libomp) caused an OpenMP clash and a
# segfault on macOS — isolation avoids that class of problem entirely.
# torch/torchvision arrive transitively via easyocr/manga-ocr; pip picks the
# right build per platform (MPS/CPU on macOS, CUDA on Linux).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo ">> creating isolated venv (.venv)"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo ">> installing pipeline dependencies (torch comes in via easyocr/manga-ocr)"
pip install -r requirements.txt

echo
echo "Setup complete. Run with:"
echo "  source .venv/bin/activate && python run.py"
