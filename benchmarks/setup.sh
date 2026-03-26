#!/usr/bin/env bash
# Creates a virtual environment and installs all benchmark dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "==> Creating virtual environment at ${VENV_DIR}"
python3 -m venv --prompt benchmarks "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "==> Setup complete!"
echo "    Activate the virtual environment with:"
echo "      source ${VENV_DIR}/bin/activate"
echo ""
echo "    Then run benchmarks, e.g.:"
echo "      cd ${SCRIPT_DIR}"
echo "      python run_all.py -m 8192 -n 8192 -k 8192 --backends triton rocblas"
