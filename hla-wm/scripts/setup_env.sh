#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "HLA-WM requires Python 3.10–3.12; set PYTHON_BIN=python3.11"'
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install --build-constraint build-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
printf '\nEnvironment ready. Activate with: source %s/bin/activate\n' "$VENV_DIR"
