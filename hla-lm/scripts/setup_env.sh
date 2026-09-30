#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
"$PYTHON_BIN" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "HLA-LM requires Python 3.11–3.13"'
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools==78.1.1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
printf '\nEnvironment ready. Activate with: source %s/bin/activate\n' "$VENV_DIR"
