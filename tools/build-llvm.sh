#!/bin/bash
set -euo pipefail

# ===----------------------------------------------------------------------=== #
# Configurable flags
# ===----------------------------------------------------------------------=== #

LLVM_PROJECT="${LLVM_PROJECT:-${HOME}/llvm-project}"
LLVM_BUILD="${LLVM_BUILD:-${HOME}/llvm-build}"
LLVM_INSTALL="${LLVM_INSTALL:-${HOME}/shared-llvm}"
LLVM_LINKER_FLAGS="${LLVM_LINKER_FLAGS:-}"
LLVM_CCACHE_BUILD="${LLVM_CCACHE_BUILD:-ON}"
LLVM_CLEAN_BUILD="${LLVM_CLEAN_BUILD:-OFF}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CC_CMD="${CC:-clang}"
CXX_CMD="${CXX:-clang++}"

# ===----------------------------------------------------------------------=== #
# Configurable but better not to touch
# ===----------------------------------------------------------------------=== #

# In Debug mode, enable assertions unconditionally.
# In Release mode, disable assertions unless explicitly overridden.
if [[ "${CMAKE_BUILD_TYPE}" == "Debug" ]]; then
  LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}"
elif [[ "${CMAKE_BUILD_TYPE}" == "Release" ]]; then
  LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-OFF}"
fi

LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-AMDGPU}"
LLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS:-mlir;lld}"
LLVM_DISTRIBUTION_COMPONENTS="\
llvm-libraries;llvm-headers;cmake-exports;\
FileCheck;llvm-config;not;count;llvm-objdump;\
mlir-libraries;mlir-headers;mlir-cmake-exports;\
mlir-opt;mlir-translate;mlir-tblgen;mlir-python-sources;\
lld;lld-headers;lld-cmake-exports;lldCommon;lldELF"
LLVM_INSTALL_UTILS="ON"
LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}"
LLVM_SRC_DIR="${LLVM_PROJECT}/llvm"

# ===----------------------------------------------------------------------=== #
# Pre-flight checks
# ===----------------------------------------------------------------------=== #

for tool in cmake ninja "${CC_CMD}" "${CXX_CMD}"; do
  if ! command -v "${tool}" &>/dev/null; then
    echo "error: required tool '${tool}' not found on PATH" >&2
    exit 1
  fi
done

if [[ ! -d "${LLVM_SRC_DIR}" ]]; then
  echo "error: LLVM source directory not found: ${LLVM_SRC_DIR}" >&2
  exit 1
fi

# ===----------------------------------------------------------------------=== #
# Configure LLVM
# ===----------------------------------------------------------------------=== #

if [[ "${LLVM_CLEAN_BUILD}" == "ON" ]]; then
  echo "Performing clean build: removing existing build directory ${LLVM_BUILD}"
  rm -rf "${LLVM_BUILD}"
fi
mkdir -p "${LLVM_BUILD}"
cmake -G Ninja \
    -S "${LLVM_SRC_DIR}" \
    -B "${LLVM_BUILD}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC_CMD}" \
    -DCMAKE_CXX_COMPILER="${CXX_CMD}" \
    -DLLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS}" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL}" \
    ${LLVM_LINKER_FLAGS} \
    -DLLVM_CCACHE_BUILD="${LLVM_CCACHE_BUILD}" \
    -DLLVM_INSTALL_UTILS="${LLVM_INSTALL_UTILS}" \
    -DLLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS}" \
    -DLLVM_APPEND_VC_REV=ON \
    -DLLVM_DISTRIBUTION_COMPONENTS="${LLVM_DISTRIBUTION_COMPONENTS}" \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LIBCXX=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DMLIR_PYTHON_STUBGEN_ENABLED=OFF \
    -DMLIR_ENABLE_PYTHON_SOURCES=ON

# ===----------------------------------------------------------------------=== #
# Build and install
# ===----------------------------------------------------------------------=== #

# Build only the distribution components.
IFS=';' read -ra DISTRIBUTION_TARGETS <<< "${LLVM_DISTRIBUTION_COMPONENTS}"
cmake --build "${LLVM_BUILD}" --target "${DISTRIBUTION_TARGETS[@]}"

# Install only the distribution components.
cmake --build "${LLVM_BUILD}" --target install-distribution
