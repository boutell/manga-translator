#!/usr/bin/env bash
# One-time setup. Creates a venv that can SEE an existing system torch
# (e.g. a ROCm build) via --system-site-packages, so we don't download or
# clobber a working GPU torch. Everything else installs into the venv.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo ">> creating venv (.venv) with access to system site-packages"
  "$PY" -m venv --system-site-packages .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo ">> checking for an existing torch (ROCm/CUDA) before installing"
if python -c "import torch" 2>/dev/null; then
  echo "   found torch $(python -c 'import torch; print(torch.__version__)') — reusing it"
else
  echo "   no system torch found — pip will install a CPU build as a dependency"
fi

echo ">> installing pipeline dependencies"
pip install -r requirements.txt

echo
echo "Setup complete. Run with:"
echo "  source .venv/bin/activate && python run.py"
