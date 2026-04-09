#!/usr/bin/env bash
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# benchmarks/setup.sh - Set up the benchmark virtual environment.
#
# Creates a venv, installs benchmark dependencies (PyTorch, Triton, etc.),
# and optionally builds aiter (AMD AI Tensor Engine) with native extensions.
#
# Safe to re-run (idempotent).  Uses uv for all Python operations.
#
# Usage:
#   bash benchmarks/setup.sh [OPTIONS]
#
# Options:
#   --skip-aiter         Skip aiter clone/build
#   --skip-requirements  Skip Python requirements installation
#   --python=PATH        Python interpreter to use
#   --venv=PATH          Use or create a specific venv (default: benchmarks/.venv)
#   --help               Show this help
set -euo pipefail

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=tools/common.sh
source "$ASTER_DIR/tools/common.sh"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

print_help() {
    echo "Usage: benchmarks/setup.sh [OPTIONS]"
    echo ""
    echo "Set up the benchmark virtual environment with PyTorch, Triton, ROCm SDK,"
    echo "and optionally aiter (AMD AI Tensor Engine)."
    echo ""
    echo "Options:"
    echo "  --rocm-target=T      Select ROCm target non-interactively (e.g. gfx94X)"
    echo "  --skip-rocm          Skip ROCm SDK installation"
    echo "  --skip-aiter         Skip aiter clone and build"
    echo "  --skip-requirements  Skip Python requirements installation"
    echo "  --python=PATH        Python interpreter to use"
    echo "  --venv=PATH          Use or create a specific venv [default: benchmarks/.venv]"
    echo "  --help               Show this help"
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKIP_AITER=false
SKIP_ROCM=false
SKIP_REQUIREMENTS=false
ROCM_TARGET_EXPLICIT=""
PYTHON_EXPLICIT=""
VENV_DIR="${SCRIPT_DIR}/.venv"

parse_arguments() {
    for arg in "$@"; do
        case "$arg" in
            --skip-aiter)        SKIP_AITER=true ;;
            --skip-rocm)         SKIP_ROCM=true ;;
            --skip-requirements) SKIP_REQUIREMENTS=true ;;
            --rocm-target=*)     ROCM_TARGET_EXPLICIT="${arg#*=}" ;;
            --python=*)          PYTHON_EXPLICIT="${arg#*=}" ;;
            --venv=*)            VENV_DIR="${arg#*=}" ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                err "Unknown option: $arg"
                echo "Run 'benchmarks/setup.sh --help' for usage."
                exit 1
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Phase 1: Prerequisites
# ---------------------------------------------------------------------------

phase1_prerequisites() {
    info "Phase 1: Checking prerequisites"
    MISSING=()

    check_required_cmd uv
    check_required_cmd git
    resolve_python

    if [ ${#MISSING[@]} -gt 0 ]; then
        err "Missing prerequisites: ${MISSING[*]}"
        exit 1
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 2: Virtual environment + requirements
# ---------------------------------------------------------------------------

phase2_venv() {
    info "Phase 2: Virtual environment"

    create_or_reuse_venv "$VENV_DIR" "benchmarks"

    if ! "$VENV_DIR/bin/python" -c "import sys" 2>/dev/null; then
        err "venv python is broken at $VENV_DIR/bin/python"
        exit 1
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Phase 3: ROCm SDK
# ---------------------------------------------------------------------------

phase3_rocm() {
    if [ "$SKIP_ROCM" = true ]; then
        info "Phase 3: ROCm SDK (skipped via --skip-rocm)"
        echo ""
        return
    fi

    if [ "$(uname)" = "Darwin" ]; then
        warn "ROCm SDK is only supported on Linux, skipping"
        echo ""
        return
    fi

    info "Phase 3: ROCm SDK"
    select_rocm_target "$ASTER_DIR" "$ROCM_TARGET_EXPLICIT"
    install_rocm_sdk "$VENV_DIR"
    configure_rocm_env "$VENV_DIR"
    init_rocm_sdk "$VENV_DIR" "false"
    patch_activate_rocm "$VENV_DIR"

    # Install Python requirements using TheRock as the extra index so that
    # ROCm-enabled PyTorch and Triton wheels are preferred over PyPI builds.
    local rocm_index_url
    rocm_index_url=$(head -1 "$ROCM_REQ" | sed 's/^-i //')
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "requirements installation skipped (--skip-requirements)"
    else
        install_requirements "$VENV_DIR" "$SCRIPT_DIR/requirements.txt" \
            "--extra-index-url $rocm_index_url"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Phase 4: aiter (AMD AI Tensor Engine)
# ---------------------------------------------------------------------------

phase4_aiter() {
    if [ "$SKIP_AITER" = true ]; then
        info "Phase 4: aiter (skipped via --skip-aiter)"
        echo ""
        return
    fi

    info "Phase 4: aiter (AMD AI Tensor Engine)"

    AITER_DIR="${SCRIPT_DIR}/aiter_src"

    # Clone if needed.
    if [ ! -d "${AITER_DIR}" ]; then
        echo "  Cloning aiter..."
        git clone https://github.com/ROCm/aiter.git "${AITER_DIR}"
        ok "aiter cloned"
    else
        ok "aiter already cloned at ${AITER_DIR}"
    fi

    echo "  Initialising aiter submodules (composable_kernel / CK)..."
    git -C "${AITER_DIR}" submodule update --init --recursive
    ok "submodules initialised"

    # Install aiter's own requirements.
    install_requirements "$VENV_DIR" "${AITER_DIR}/requirements.txt"

    # Build native extensions (requires hipcc / ROCm in PATH).
    echo "  Building aiter native extensions (requires hipcc / ROCm)..."
    if uv pip install --python "$VENV_DIR/bin/python" -e "${AITER_DIR}"; then
        ok "aiter installed"
    else
        err "aiter build failed (is ROCm available?)"
        exit 1
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    info "Setup complete!"
    echo ""
    echo "  venv:  $VENV_DIR"
    echo ""
    echo "  Activate the virtual environment:"
    echo "    source ${VENV_DIR}/bin/activate"
    echo ""
    echo "  Then run benchmarks, e.g.:"
    echo "    cd ${SCRIPT_DIR}"
    echo "    python run_all.py -m 8192 -n 8192 -k 8192 --backends triton rocblas"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    parse_arguments "$@"

    phase1_prerequisites
    phase2_venv
    phase3_rocm
    phase4_aiter
    print_summary
}

main "$@"
