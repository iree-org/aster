#!/usr/bin/env bash
# linux/entrypoint.sh - Build and package LLVM inside a manylinux container.
#
# Expected env vars (set by build.sh via `docker run -e ...`):
#   LLVM_COMMIT     Full commit SHA to build.
#   JOBS            Number of parallel build jobs.
#   OUTPUT_DIR      Host-mounted directory to write the final tar.gz into.
#   ASTER_SRC       Path to the aster source tree (inside or mounted into the
#                   container), used to locate tools/build-llvm.sh and
#                   tools/pkgs/llvm/common/cpack-config.cmake.in.
#                   Defaults to /pkgs (the COPY destination in the Dockerfile).
set -euo pipefail

ASTER_SRC="${ASTER_SRC:-/pkgs}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
JOBS="${JOBS:-$(nproc)}"
LLVM_COMMIT="${LLVM_COMMIT:?LLVM_COMMIT must be set}"

LLVM_COMMIT_SHORT="${LLVM_COMMIT:0:7}"
ARCH="$(uname -m)"   # x86_64 or aarch64
LLVM_OS="linux"

WORK_DIR="$(mktemp -d)"
# Use a caller-supplied LLVM source tree to skip the ~500 MB clone.
LLVM_PROJECT="${LLVM_SRC_DIR:-${WORK_DIR}/llvm-project}"
LLVM_BUILD="${WORK_DIR}/llvm-build"
LLVM_INSTALL="${WORK_DIR}/llvm-install"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info() { echo "==> $*"; }

# ---------------------------------------------------------------------------
# 1. Shallow-clone LLVM at the pinned commit (skip if a checkout was mounted).
# ---------------------------------------------------------------------------

if [[ -d "${LLVM_PROJECT}/.git" ]]; then
    info "Reusing existing checkout at ${LLVM_PROJECT}."
else
    info "Cloning llvm-project at ${LLVM_COMMIT_SHORT}..."
    git init "${LLVM_PROJECT}"
    git -C "${LLVM_PROJECT}" remote add origin \
        https://github.com/nicolasvasilache/llvm-project.git
    git -C "${LLVM_PROJECT}" fetch --depth 1 origin "${LLVM_COMMIT}"
    git -C "${LLVM_PROJECT}" checkout FETCH_HEAD
fi

# ---------------------------------------------------------------------------
# 2. Build LLVM using the existing build-llvm.sh.
#    Pass -static-libstdc++ / -static-libgcc so the resulting ELF binaries
#    depend only on glibc, with no libstdc++ runtime requirement.
# ---------------------------------------------------------------------------

info "Building LLVM (${JOBS} jobs)..."
LLVM_PROJECT="${LLVM_PROJECT}" \
LLVM_BUILD="${LLVM_BUILD}" \
LLVM_INSTALL="${LLVM_INSTALL}" \
LLVM_ENABLE_ASSERTIONS=ON \
LLVM_CCACHE_BUILD=OFF \
CMAKE_BUILD_TYPE=Release \
CMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
NINJA_STATUS="[%f/%t %es] " \
    bash "${ASTER_SRC}/tools/build-llvm.sh"

# ---------------------------------------------------------------------------
# 3. Fix RPATHs on any ELF shared objects that ended up in lib/.
#    Executables are fully statically linked against C++ runtime, so this is
#    a safety net for unexpected .so files (e.g. Python extension modules).
# ---------------------------------------------------------------------------

info "Patching RPATHs..."
find "${LLVM_INSTALL}" -type f \( -name "*.so" -o -name "*.so.*" \) | \
while read -r lib; do
    patchelf --set-rpath '$ORIGIN' "${lib}" || true
done
find "${LLVM_INSTALL}/bin" -type f | xargs -I{} sh -c \
    'file "$1" | grep -q ELF && patchelf --set-rpath '"'"'$ORIGIN/../lib'"'"' "$1" || true' \
    -- {}

# ---------------------------------------------------------------------------
# 4. Package with CPack TGZ.
#    Fill the .in template and run cpack to produce the archive.
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
