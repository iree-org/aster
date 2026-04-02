#!/usr/bin/env bash
# tools/pkgs/aster/build.sh - Build the aster Python wheel.
#
# Usage:
#   bash tools/pkgs/aster/build.sh [OPTIONS]
#
# Options:
#   --llvm-tarball=PATH  Path to a pre-built LLVM tar.gz.
#   --llvm-dir=PATH      Path to an already-unpacked LLVM installation.
#   --output=DIR         Write the resulting wheel(s) here.  [default: ./wheelhouse]
#   --arch=ARCH          Target architecture (x86_64 | aarch64 | arm64).
#                        [default: host architecture; always passed to cibuildwheel
#                         to prevent unintentional cross-arch builds]
#   --jobs=N             Parallelism forwarded to CMake via CMAKE_BUILD_PARALLEL_LEVEL.
#   --help               Show this message.
#
# Exactly one of --llvm-tarball or --llvm-dir must be provided.
#
# The LLVM tarball format must match the output of tools/pkgs/llvm/build.sh:
# a tar.gz with a top-level llvm/ directory containing bin/, include/, lib/.
# When --llvm-dir is used, build.sh re-archives the directory in that format.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTER_SRC="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUTPUT_DIR="$(pwd)/wheelhouse"
LLVM_TARBALL_ARG=""
LLVM_DIR_ARG=""
ARCH="$(uname -m)"
JOBS=""

print_help() {
    sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "${BASH_SOURCE[0]}"
}

for arg in "$@"; do
    case "${arg}" in
        --llvm-tarball=*) LLVM_TARBALL_ARG="${arg#--llvm-tarball=}" ;;
        --llvm-dir=*)     LLVM_DIR_ARG="${arg#--llvm-dir=}" ;;
        --output=*)       OUTPUT_DIR="${arg#--output=}" ;;
        --arch=*)         ARCH="${arg#--arch=}" ;;
        --jobs=*)         JOBS="${arg#--jobs=}" ;;
        --help)           print_help; exit 0 ;;
        *) echo "error: unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

# Expand leading ~ in paths (bash does not expand ~ inside substitutions).
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
LLVM_TARBALL_ARG="${LLVM_TARBALL_ARG/#\~/$HOME}"
LLVM_DIR_ARG="${LLVM_DIR_ARG/#\~/$HOME}"

_llvm_sources=0
[[ -n "${LLVM_TARBALL_ARG}" ]] && _llvm_sources=$((_llvm_sources + 1))
[[ -n "${LLVM_DIR_ARG}" ]]     && _llvm_sources=$((_llvm_sources + 1))

if [[ "${_llvm_sources}" -eq 0 ]]; then
    echo "error: one of --llvm-tarball or --llvm-dir is required" >&2
    exit 1
fi
if [[ "${_llvm_sources}" -gt 1 ]]; then
    echo "error: --llvm-tarball and --llvm-dir are mutually exclusive" >&2
    exit 1
fi

if ! command -v cibuildwheel &>/dev/null; then
    echo "error: cibuildwheel not found on PATH" >&2
    echo "Install it with: pip install cibuildwheel" >&2
    exit 1
fi

if [[ "$(uname -s)" == "Linux" ]]; then
    if ! command -v docker &>/dev/null; then
        echo "error: docker is required for Linux wheel building" >&2
        exit 1
    fi
fi

# All LLVM input modes converge on a single staged tarball.  The before-all.sh
# hook extracts it via the LLVM_PKG_PATH env var.
STAGED_TARBALL="${ASTER_SRC}/.llvm-pkg.tar.gz"
_STAGED=0

cleanup() {
    if [[ "${_STAGED}" -eq 1 && -f "${STAGED_TARBALL}" ]]; then
        rm -f "${STAGED_TARBALL}"
    fi
}
trap cleanup EXIT

if [[ -n "${LLVM_TARBALL_ARG}" ]]; then
    LLVM_TARBALL_ARG="$(cd "$(dirname "${LLVM_TARBALL_ARG}")" && pwd)/$(basename "${LLVM_TARBALL_ARG}")"
    if [[ "${LLVM_TARBALL_ARG}" == "${ASTER_SRC}/"* ]]; then
        STAGED_TARBALL="${LLVM_TARBALL_ARG}"
        echo "==> Using in-tree LLVM tarball: ${STAGED_TARBALL}"
    else
        echo "==> Copying LLVM tarball into project tree ..."
        cp "${LLVM_TARBALL_ARG}" "${STAGED_TARBALL}"
        _STAGED=1
    fi

elif [[ -n "${LLVM_DIR_ARG}" ]]; then
    LLVM_DIR_ARG="$(realpath "${LLVM_DIR_ARG}")"
    if [[ ! -d "${LLVM_DIR_ARG}" ]]; then
        echo "error: LLVM directory not found: ${LLVM_DIR_ARG}" >&2
        exit 1
    fi
    echo "==> Re-archiving LLVM directory ${LLVM_DIR_ARG} ..."
    # Symlink as llvm/ so the archive has a top-level llvm/ directory,
    # matching the layout expected by before-all.sh.
    _TMP_STAGE="$(mktemp -d)"
    ln -s "${LLVM_DIR_ARG}" "${_TMP_STAGE}/llvm"
    tar -hczf "${STAGED_TARBALL}" -C "${_TMP_STAGE}" llvm
    rm -rf "${_TMP_STAGE}"
    _STAGED=1
fi

LLVM_PKG_PATH="${STAGED_TARBALL#${ASTER_SRC}/}"

echo "==> LLVM package path : ${LLVM_PKG_PATH}"
echo "==> Output directory  : ${OUTPUT_DIR}"
echo "==> Target arch       : ${ARCH}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

CIBW_ARGS=(--output-dir "${OUTPUT_DIR}")
[[ -n "${ARCH}" ]] && CIBW_ARGS+=(--archs "${ARCH}")

echo "==> Running cibuildwheel ..."
(
    cd "${ASTER_SRC}"
    LLVM_PKG_PATH="${LLVM_PKG_PATH}" \
        ${JOBS:+CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"} \
        cibuildwheel "${CIBW_ARGS[@]}" .
)

echo "==> Wheel(s) written to ${OUTPUT_DIR}."
