#!/usr/bin/env bash
# macos/build.sh - Build and package LLVM on macOS.
#
# Called by tools/pkgs/llvm/build.sh when running on macOS.
# Requires: cmake, ninja, clang, git, pip (for delocate).
#
# Expected env vars:
#   LLVM_COMMIT     Full commit SHA to build.
#   JOBS            Number of parallel build jobs.
#   OUTPUT_DIR      Directory to write the final tar.gz into.
#   ASTER_SRC       Root of the aster source tree (contains tools/build-llvm.sh).
#   LLVM_SRC_DIR    (optional) Reuse an existing llvm-project checkout.
set -euo pipefail

ASTER_SRC="${ASTER_SRC:?ASTER_SRC must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu)}"
LLVM_COMMIT="${LLVM_COMMIT:?LLVM_COMMIT must be set}"

LLVM_COMMIT_SHORT="${LLVM_COMMIT:0:7}"
ARCH="$(uname -m)"   # arm64 or x86_64
LLVM_OS="macos"

WORK_DIR="$(mktemp -d)"
LLVM_PROJECT="${LLVM_SRC_DIR:-${WORK_DIR}/llvm-project}"
LLVM_BUILD="${WORK_DIR}/llvm-build"
LLVM_INSTALL="${WORK_DIR}/llvm-install"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info() { echo "==> $*"; }

# ---------------------------------------------------------------------------
# 1. Shallow-clone LLVM (skip if caller supplied an existing checkout).
# ---------------------------------------------------------------------------

if [[ ! -d "${LLVM_PROJECT}/.git" ]]; then
    info "Cloning llvm-project at ${LLVM_COMMIT_SHORT}..."
    git init "${LLVM_PROJECT}"
    git -C "${LLVM_PROJECT}" remote add origin \
        https://github.com/nicolasvasilache/llvm-project.git
    git -C "${LLVM_PROJECT}" fetch --depth 1 origin "${LLVM_COMMIT}"
    git -C "${LLVM_PROJECT}" checkout FETCH_HEAD
else
    info "Reusing existing checkout at ${LLVM_PROJECT}."
fi

# ---------------------------------------------------------------------------
# 2. Build LLVM.
#    Set MACOSX_DEPLOYMENT_TARGET so binaries run on macOS 12+.
#    libc++ is always available as a system library so no bundling needed.
# ---------------------------------------------------------------------------

info "Building LLVM (${JOBS} jobs)..."
LLVM_PROJECT="${LLVM_PROJECT}" \
LLVM_BUILD="${LLVM_BUILD}" \
LLVM_INSTALL="${LLVM_INSTALL}" \
LLVM_ENABLE_ASSERTIONS=ON \
LLVM_CCACHE_BUILD=OFF \
CMAKE_BUILD_TYPE=Release \
MACOSX_DEPLOYMENT_TARGET=12.0 \
NINJA_STATUS="[%f/%t %es] " \
    bash "${ASTER_SRC}/tools/build-llvm.sh"

# ---------------------------------------------------------------------------
# 3. Bundle non-system dylibs with delocate.
#    System libraries (/usr/lib/libc++.dylib, libSystem, etc.) are excluded
#    automatically by delocate.
# ---------------------------------------------------------------------------

if ! command -v delocate-path &>/dev/null; then
    info "Installing delocate..."
    pip install --quiet delocate
fi

info "Running delocate-path..."
delocate-path "${LLVM_INSTALL}/bin" "${LLVM_INSTALL}/lib" \
    --lib-path "${LLVM_INSTALL}/lib" || true

# ---------------------------------------------------------------------------
# 4. Package with CPack TGZ.
# ---------------------------------------------------------------------------

CPACK_WORK="${WORK_DIR}/cpack"
mkdir -p "${CPACK_WORK}"

CPACK_IN="${ASTER_SRC}/tools/pkgs/llvm/common/cpack-config.cmake.in"
CPACK_CFG="${CPACK_WORK}/cpack-config.cmake"

sed \
    -e "s|@LLVM_COMMIT_SHORT@|${LLVM_COMMIT_SHORT}|g" \
    -e "s|@LLVM_OS@|${LLVM_OS}|g" \
    -e "s|@LLVM_ARCH@|${ARCH}|g" \
    -e "s|@LLVM_INSTALL_DIR@|${LLVM_INSTALL}|g" \
    "${CPACK_IN}" > "${CPACK_CFG}"

info "Packaging with CPack..."
cpack --config "${CPACK_CFG}" -B "${CPACK_WORK}"

ARCHIVE="${CPACK_WORK}/llvm-${LLVM_OS}-${ARCH}-${LLVM_COMMIT_SHORT}.tar.gz"
cp "${ARCHIVE}" "${OUTPUT_DIR}/"

info "Done: ${OUTPUT_DIR}/$(basename "${ARCHIVE}")"
