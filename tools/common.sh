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
