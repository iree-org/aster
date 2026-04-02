#!/usr/bin/env bash
# tools/pkgs/llvm/build.sh - Build and package LLVM for distribution.
#
# Produces a relocatable tar.gz containing LLVM binaries, headers, and CMake
# exports.  Users extract the archive and point CMAKE_PREFIX_PATH at the
# resulting llvm/ directory to use it in their builds.
#
# Usage:
#   bash tools/pkgs/llvm/build.sh [OPTIONS]
#
# Options:
#   --output=DIR       Write the output tar.gz here.          [default: .]
#   --llvm-src=DIR     Reuse an existing llvm-project clone.  [default: clone]
#   --jobs=N           Parallel build jobs.  [default: nproc / hw.logicalcpu]
#   --arch=ARCH        Linux only: x86_64 or aarch64.         [default: host]
#   --help             Show this message.
#
# Environment variables:
#   LLVM_COMMIT   Override the commit to build.  Defaults to the contents of
#                 llvm/LLVM_COMMIT in the aster source tree.
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the aster source tree (the directory that contains tools/).
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTER_SRC="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OUTPUT_DIR="$(pwd)"
LLVM_SRC_DIR=""
ARCH="$(uname -m)"
case "$(uname -s)" in
    Darwin) JOBS="$(sysctl -n hw.logicalcpu)" ;;
    *)      JOBS="$(nproc)" ;;
esac

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

print_help() {
    sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "${BASH_SOURCE[0]}"
}

for arg in "$@"; do
    case "${arg}" in
        --output=*)   OUTPUT_DIR="${arg#--output=}" ;;
        --llvm-src=*) LLVM_SRC_DIR="${arg#--llvm-src=}" ;;
        --jobs=*)     JOBS="${arg#--jobs=}" ;;
        --arch=*)     ARCH="${arg#--arch=}" ;;
        --help)       print_help; exit 0 ;;
        *) echo "error: unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Read the pinned LLVM commit from the aster source tree.
# ---------------------------------------------------------------------------

LLVM_COMMIT_FILE="${ASTER_SRC}/llvm/LLVM_COMMIT"
if [[ -z "${LLVM_COMMIT:-}" ]]; then
    if [[ ! -f "${LLVM_COMMIT_FILE}" ]]; then
        echo "error: cannot find ${LLVM_COMMIT_FILE}" >&2
        exit 1
    fi
    LLVM_COMMIT="$(head -1 "${LLVM_COMMIT_FILE}" | tr -d '[:space:]')"
fi

# Expand leading ~ to $HOME (bash does not expand ~ inside parameter substitutions).
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
LLVM_SRC_DIR="${LLVM_SRC_DIR/#\~/$HOME}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
[[ -n "${LLVM_SRC_DIR}" ]] && LLVM_SRC_DIR="$(cd "${LLVM_SRC_DIR}" && pwd)"

echo "==> LLVM commit : ${LLVM_COMMIT}"
echo "==> Output dir  : ${OUTPUT_DIR}"
echo "==> Jobs        : ${JOBS}"

# ---------------------------------------------------------------------------
# Dispatch to the platform-specific build script.
# ---------------------------------------------------------------------------

OS="$(uname -s)"

case "${OS}" in
    # -----------------------------------------------------------------------
    Linux)
        if ! command -v docker &>/dev/null; then
            echo "error: docker is required for Linux packaging" >&2
            exit 1
        fi

        case "${ARCH}" in
            x86_64)  MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"  ;;
            aarch64) MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_aarch64" ;;
            *) echo "error: unsupported arch for Linux packaging: ${ARCH}" >&2; exit 1 ;;
        esac

        echo "==> Building Docker image (base: ${MANYLINUX_IMAGE})..."
        docker build \
            --build-arg "MANYLINUX_IMAGE=${MANYLINUX_IMAGE}" \
            -t llvm-pkg-builder \
            -f "${SCRIPT_DIR}/linux/Dockerfile" \
            "${ASTER_SRC}/tools/pkgs/llvm"

        DOCKER_ARGS=(
            --rm
            -e "LLVM_COMMIT=${LLVM_COMMIT}"
            -e "JOBS=${JOBS}"
            -e "OUTPUT_DIR=/output"
            -e "ASTER_SRC=/aster-src"
            -v "${OUTPUT_DIR}:/output"
            -v "${ASTER_SRC}:/aster-src:ro"
        )

        # Mount an existing LLVM checkout if provided (avoids re-cloning).
        if [[ -n "${LLVM_SRC_DIR}" ]]; then
            DOCKER_ARGS+=(-e "LLVM_SRC_DIR=/llvm-src" -v "${LLVM_SRC_DIR}:/llvm-src:ro")
        fi

        echo "==> Running build inside Docker container..."
        docker run "${DOCKER_ARGS[@]}" llvm-pkg-builder
        ;;

    # -----------------------------------------------------------------------
    Darwin)
        for tool in cmake ninja git; do
            command -v "${tool}" &>/dev/null || \
                { echo "error: required tool '${tool}' not found on PATH" >&2; exit 1; }
        done

        LLVM_SRC_DIR="${LLVM_SRC_DIR}" \
        LLVM_COMMIT="${LLVM_COMMIT}" \
        JOBS="${JOBS}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        ASTER_SRC="${ASTER_SRC}" \
            bash "${SCRIPT_DIR}/macos/build.sh"
        ;;

    *)
        echo "error: unsupported platform: ${OS}" >&2
        exit 1
        ;;
esac
