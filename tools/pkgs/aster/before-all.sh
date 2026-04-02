#!/usr/bin/env bash
# before-all.sh - cibuildwheel before-all hook for the aster Python wheel.
#
# Extracts the LLVM tarball and installs build tools.  Runs once per build
# environment before any Python version is built.
#
# Required env var:
#   LLVM_PKG_PATH  Path to the LLVM tar.gz, relative to the project root.
set -euo pipefail

OS="$(uname -s)"

PROJECT_DIR="$(pwd)"
LLVM_TARBALL="${PROJECT_DIR}/${LLVM_PKG_PATH:?LLVM_PKG_PATH must be set}"

if [[ ! -f "${LLVM_TARBALL}" ]]; then
  echo "error: LLVM tarball not found: ${LLVM_TARBALL}" >&2
  exit 1
fi

if [[ "${OS}" == "Linux" ]]; then
  LLVM_INSTALL="/opt/llvm"
else
  LLVM_INSTALL="${HOME}/.llvm-pkg"
fi

# Extract to a stable directory and locate lib/cmake, then symlink
# LLVM_INSTALL -> lib/cmake so that CMAKE_PREFIX_PATH resolves
# <PackageName>/Config.cmake regardless of the tarball's internal layout.
# The extract directory must persist for the lifetime of the build since
# LLVM_INSTALL is a symlink into it.
_EXTRACT_DIR="${PROJECT_DIR}/.llvm-extract"
rm -rf "${_EXTRACT_DIR}"
mkdir -p "${_EXTRACT_DIR}"

echo "Extracting ${LLVM_TARBALL} ..."
tar -xzf "${LLVM_TARBALL}" -C "${_EXTRACT_DIR}"

_CMAKE_DIR="$(set +o pipefail; find "${_EXTRACT_DIR}" -maxdepth 8 -type d -name "cmake" -path "*/lib/cmake" | head -n1)"
if [[ -z "${_CMAKE_DIR}" ]]; then
  echo "error: lib/cmake directory not found in tarball" >&2
  exit 1
fi
echo "LLVM cmake dir: ${_CMAKE_DIR}"

ln -sfn "${_CMAKE_DIR}" "${LLVM_INSTALL}"
echo "LLVM cmake dir linked at ${LLVM_INSTALL}."

if [[ "${OS}" == "Linux" ]]; then
  echo "Installing build tools ..."
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y clang lld cmake ninja-build
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache clang lld cmake ninja
  fi
  echo "Installing nanobind ..."
  pip install "nanobind>=2.9,<3.0"
elif [[ "${OS}" == "Darwin" ]]; then
  for _tool in cmake ninja clang; do
    if ! command -v "${_tool}" >/dev/null 2>&1; then
      echo "error: required tool not found: ${_tool}" >&2
      echo "Install it via Homebrew: brew install ${_tool}" >&2
      exit 1
    fi
  done
  echo "Installing nanobind ..."
  pip install "nanobind>=2.9,<3.0"
fi

echo "before-all.sh complete."
