#!/usr/bin/env bash
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# tools/common.sh - Shared shell helpers for ASTER setup scripts.
#
# Source this file; do not execute it directly.
#   source "$(dirname "$0")/../tools/common.sh"   # from benchmarks/
#   source "$(dirname "$0")/common.sh"             # from tools/

# Guard against double-sourcing.
[ -n "${_ASTER_COMMON_SH:-}" ] && return 0
_ASTER_COMMON_SH=1

# ---------------------------------------------------------------------------
# Colour helpers (disabled when NO_COLOR is set or stdout is not a terminal)
# ---------------------------------------------------------------------------

if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then
    RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
else
    RED="\033[0;31m" GREEN="\033[0;32m" YELLOW="\033[0;33m"
    BLUE="\033[0;34m" BOLD="\033[1m" RESET="\033[0m"
fi

info()  { echo -e "${BLUE}==> ${RESET}${BOLD}$*${RESET}"; }
ok()    { echo -e "${GREEN} OK ${RESET}$*"; }
warn()  { echo -e "${YELLOW}WARN${RESET} $*"; }
err()   { echo -e "${RED}FAIL${RESET} $*"; }
ask()   {
    echo -en "${YELLOW}?${RESET} $* [y/N] "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Prerequisite checking
# ---------------------------------------------------------------------------

# MISSING is a global array; callers should initialise it before use.
MISSING=()

add_missing() {
    local item="$1"
    local existing
    for existing in "${MISSING[@]}"; do
        [ "$existing" = "$item" ] && return
    done
    MISSING+=("$item")
}

check_required_cmd() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd ($(command -v "$cmd"))"
    else
        err "$cmd not found"
        add_missing "$cmd"
    fi
}

check_optional_cmd() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd ($(command -v "$cmd"))"
    else
        warn "$cmd not found (optional)"
    fi
}

# ---------------------------------------------------------------------------
# Python / venv helpers
# ---------------------------------------------------------------------------

# Locate a suitable Python >= 3.12.  Sets PYTHON on success, adds to MISSING
# on failure.  Callers may pre-set PYTHON_EXPLICIT to override auto-detection.
resolve_python() {
    if [ -n "${PYTHON_EXPLICIT:-}" ]; then
        if ! command -v "$PYTHON_EXPLICIT" >/dev/null 2>&1; then
            err "specified python not found: $PYTHON_EXPLICIT"
            add_missing "$PYTHON_EXPLICIT"
            PYTHON=""
            return
        fi
        PYTHON="$PYTHON_EXPLICIT"
        PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        ok "python $PY_VERSION ($PYTHON) [--python]"
        return
    fi

    PYTHON=""
    if command -v uv >/dev/null 2>&1; then
        PYTHON=$(uv python find 3.12 2>/dev/null || true)
        if [ -n "$PYTHON" ]; then
            ok "python 3.12 via uv ($PYTHON)"
        fi
    fi

    if [ -z "$PYTHON" ] && command -v python3 >/dev/null 2>&1; then
        PYTHON=$(python3 -c "import sys; print(sys.executable)")
        if python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"; then
            PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            ok "python3 $PY_VERSION ($PYTHON)"
        else
            PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            err "python3 version $PY_VERSION is too old (need >= 3.12)"
            add_missing "python3>=3.12"
            PYTHON=""
        fi
    fi

    if [ -z "$PYTHON" ]; then
        err "No suitable python found"
        add_missing "python3>=3.12"
    fi
}

# Create (or reuse) a venv via uv.
#   $1 - venv path
#   $2 - prompt name (e.g. "aster", "benchmarks")
#   Requires PYTHON to be set (call resolve_python first).
create_or_reuse_venv() {
    local venv_dir="$1"
    local prompt="${2:-venv}"

    if [ -f "$venv_dir/bin/python" ]; then
        ok "venv exists at $venv_dir"
        return
    fi

    echo "  Creating venv at $venv_dir with $PYTHON..."
    if ! uv venv "$venv_dir" --seed --python "$PYTHON" --prompt "$prompt"; then
        err "Failed to create Python venv"
        exit 1
    fi
    ok "venv created"
}

# Install a requirements file via uv pip (with stamp-based caching).
#   $1 - venv path
#   $2 - requirements.txt path
#   $3 - (optional) extra uv pip flags, e.g. "--prerelease=allow"
install_requirements() {
    local venv_dir="$1"
    local req_file="$2"
    local extra_flags="${3:-}"

    local stamp="$venv_dir/.requirements-stamp-$(basename "$req_file")"
    if [ -f "$stamp" ] && [ "$stamp" -nt "$req_file" ]; then
        ok "requirements up to date ($(basename "$req_file"))"
        return
    fi

    # uv pip doesn't support --pre inside requirements files; translate it
    # to the CLI equivalent --prerelease=allow automatically.
    if grep -q '^\s*--pre\b' "$req_file" 2>/dev/null; then
        extra_flags="--prerelease=allow ${extra_flags}"
    fi

    echo "  Installing $(basename "$req_file")..."
    # shellcheck disable=SC2086
    if uv pip install --python "$venv_dir/bin/python" -r "$req_file" $extra_flags; then
        touch "$stamp"
        ok "$(basename "$req_file") installed"
    else
        err "Failed to install $(basename "$req_file")"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# ROCm SDK helpers
# ---------------------------------------------------------------------------

# Select which requirements-amd-*.txt to use.
# Sets ROCM_REQ (full path) and ROCM_TARGET (e.g. "gfx94X").
#   $1 - repo root containing requirements-amd-*.txt files
#   $2 - (optional) explicit target, e.g. "gfx94X"
select_rocm_target() {
    local repo_dir="$1"
    local explicit="${2:-}"

    ROCM_REQ_FILES=()
    for f in "$repo_dir"/requirements-amd-*.txt; do
        [ -f "$f" ] && ROCM_REQ_FILES+=("$f")
    done

    if [ ${#ROCM_REQ_FILES[@]} -eq 0 ]; then
        err "No requirements-amd-*.txt files found in $repo_dir"
        exit 1
    fi

    if [ -n "$explicit" ]; then
        ROCM_REQ="$repo_dir/requirements-amd-$explicit.txt"
        if [ ! -f "$ROCM_REQ" ]; then
            err "Unknown ROCm target: $explicit"
            exit 1
        fi
        ROCM_TARGET="$explicit"
        return
    fi

    if [ ${#ROCM_REQ_FILES[@]} -eq 1 ]; then
        ROCM_REQ="${ROCM_REQ_FILES[0]}"
        ROCM_TARGET=$(basename "$ROCM_REQ" .txt)
        ROCM_TARGET=${ROCM_TARGET#requirements-amd-}
        ok "Using only available ROCm target: $ROCM_TARGET"
        return
    fi

    if [ ! -t 0 ]; then
        err "Cannot prompt for ROCm target in non-interactive mode"
        exit 1
    fi

    echo ""
    echo "  Available ROCm SDK targets:"
    for i in "${!ROCM_REQ_FILES[@]}"; do
        local basename_noext
        basename_noext=$(basename "${ROCM_REQ_FILES[$i]}" .txt)
        local target=${basename_noext#requirements-amd-}
        echo "    $((i+1))) $target"
    done
    echo ""
    echo -n "  Which target? [1-${#ROCM_REQ_FILES[@]}] "
    read -r ROCM_CHOICE

    if ! [[ "$ROCM_CHOICE" =~ ^[0-9]+$ ]] || [ "$ROCM_CHOICE" -lt 1 ] || [ "$ROCM_CHOICE" -gt ${#ROCM_REQ_FILES[@]} ]; then
        err "Invalid choice: $ROCM_CHOICE"
        exit 1
    fi

    ROCM_REQ="${ROCM_REQ_FILES[$((ROCM_CHOICE-1))]}"
    ROCM_TARGET=$(basename "$ROCM_REQ" .txt)
    ROCM_TARGET=${ROCM_TARGET#requirements-amd-}
}

# Install ROCm SDK via uv pip (with stamp-based caching).
#   $1 - venv path
# Requires ROCM_REQ and ROCM_TARGET to be set (call select_rocm_target first).
install_rocm_sdk() {
    local venv_dir="$1"

    info "Installing ROCm SDK for $ROCM_TARGET"

    local stamp="$venv_dir/.rocm-stamp-$ROCM_TARGET"
    if [ -f "$stamp" ] && [ "$stamp" -nt "$ROCM_REQ" ]; then
        ok "ROCm SDK ($ROCM_TARGET) already installed"
        return
    fi

    echo "  Installing ROCm SDK from $(head -1 "$ROCM_REQ" | sed 's/-i //')..."
    echo "  This downloads ~2 GB of AMD GPU libraries."
    echo ""
    if uv pip install --python "$venv_dir/bin/python" -r "$ROCM_REQ"; then
        rm -f "$venv_dir"/.rocm-stamp-* 2>/dev/null
        touch "$stamp"
        ok "ROCm SDK ($ROCM_TARGET) installed"
    else
        err "Failed to install ROCm SDK"
        exit 1
    fi
}

# Set ROCM_PATH, HIP_PATH, and PATH to point at the venv's ROCm SDK.
#   $1 - venv path
# Sets ROCM_DEVEL as a side effect.
configure_rocm_env() {
    local venv_dir="$1"

    ROCM_DEVEL=$("$venv_dir/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel
    export ROCM_PATH="$ROCM_DEVEL"
    export HIP_PATH="$ROCM_DEVEL"
    CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/rocm' | tr '\n' ':' | sed 's/:$//')
    export PATH="$ROCM_DEVEL/bin:$CLEAN_PATH"
    ok "Isolated from system ROCm (ROCM_PATH=$ROCM_DEVEL)"
}

# Run rocm-sdk init and optionally test.
#   $1 - venv path
#   $2 - "true" to run rocm-sdk test, anything else to skip
init_rocm_sdk() {
    local venv_dir="$1"
    local do_test="${2:-false}"

    echo "  Initializing ROCm SDK..."
    if ! "$venv_dir/bin/rocm-sdk" init; then
        err "rocm-sdk init failed"
        exit 1
    fi
    ok "rocm-sdk initialized"

    if [ "$do_test" = "true" ]; then
        echo "  Testing ROCm SDK..."
        if ! "$venv_dir/bin/rocm-sdk" test; then
            err "rocm-sdk test failed"
            exit 1
        fi
        ok "rocm-sdk test passed"
    else
        ok "rocm-sdk test skipped"
    fi
}

# Patch a venv's activate script to put ROCm SDK on PATH and LD_LIBRARY_PATH.
#   $1 - venv path
# Idempotent: skips if already patched.
patch_activate_rocm() {
    local venv_dir="$1"
    local activate="$venv_dir/bin/activate"
    local marker="# --- ROCm SDK setup (added by ASTER) ---"

    if grep -q "ROCm SDK setup" "$activate" 2>/dev/null; then
        ok "activate script already has ROCm paths"
        return
    fi

    echo "  Adding ROCm SDK paths to activate script..."
    cat >> "$activate" << 'ROCM_EOF'

# --- ROCm SDK setup (added by ASTER) ---
_VENV_PURELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export ROCM_PATH="${_VENV_PURELIB}/_rocm_sdk_devel"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _VENV_PURELIB
# --- end ROCm SDK setup ---
ROCM_EOF

    ok "activate script updated with ROCm paths"
}
