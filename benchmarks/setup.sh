#!/usr/bin/env bash
# Creates a virtual environment and installs all benchmark dependencies,
# including aiter (AMD AI Tensor Engine) and its native extensions.
#
# Usage:
#   bash setup.sh [--rocm-lib PATH]
#
# Options:
#   --rocm-lib PATH   Path to the ROCm lib directory containing libamdhip64.so
#                     (default: auto-detected from ROCM_PATH, /opt/rocm/lib,
#                      or /opt/rocm-7.2.1/lib)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ROCM_LIB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-lib)
            ROCM_LIB="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-detect ROCm lib directory if not provided.
if [ -z "${ROCM_LIB}" ]; then
    if [ -n "${ROCM_PATH:-}" ] && [ -d "${ROCM_PATH}/lib" ]; then
        ROCM_LIB="${ROCM_PATH}/lib"
    elif [ -d "/opt/rocm/lib" ]; then
        ROCM_LIB="/opt/rocm/lib"
    elif [ -d "/opt/rocm-7.2.1/lib" ]; then
        ROCM_LIB="/opt/rocm-7.2.1/lib"
    else
        echo "[WARN] Could not detect ROCm lib directory; pass --rocm-lib if aiter build fails"
    fi
fi

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------

if [ -f "${VENV_DIR}/bin/activate" ]; then
    echo "==> Virtual environment already exists at ${VENV_DIR}, skipping creation"
else
    echo "==> Creating virtual environment at ${VENV_DIR}"
    python3 -m venv --prompt benchmarks "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# aiter (AMD AI Tensor Engine)
# ---------------------------------------------------------------------------

AITER_DIR="${SCRIPT_DIR}/aiter_src"
if [ ! -d "${AITER_DIR}" ]; then
    echo "==> Cloning aiter"
    git clone https://github.com/ROCm/aiter.git "${AITER_DIR}"
else
    echo "==> aiter already cloned at ${AITER_DIR}, skipping clone"
fi

echo "==> Initialising aiter submodules (composable_kernel / CK)"
git -C "${AITER_DIR}" submodule update --init --recursive

echo "==> Installing aiter Python dependencies"
pip install -r "${AITER_DIR}/requirements.txt"

echo "==> Building aiter native extensions (requires hipcc / ROCm)"
if [ -n "${ROCM_LIB}" ]; then
    echo "    ROCm lib: ${ROCM_LIB}"
    export LIBRARY_PATH="${ROCM_LIB}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
pip install -e "${AITER_DIR}"

echo ""
echo "==> Setup complete!"
echo "    Activate the virtual environment with:"
echo "      source ${VENV_DIR}/bin/activate"
echo ""
echo "    Then run benchmarks, e.g.:"
echo "      cd ${SCRIPT_DIR}"
echo "      python run_all.py -m 8192 -n 8192 -k 8192 --backends triton rocblas"
