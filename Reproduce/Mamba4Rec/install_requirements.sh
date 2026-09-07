#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$PYTHON_BIN" -c \
  'import sys, torch; sys.path.insert(0, "'"$PROJECT_ROOT"'"); from model import MAMBA_BACKEND; print(f"torch={torch.__version__} cuda={torch.version.cuda} mamba_backend={MAMBA_BACKEND}")'
