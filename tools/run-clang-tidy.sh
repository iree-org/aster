#!/bin/bash
# Run clang-tidy on ASTER using the same approach as IREE:
#   - Default clang-tidy checks (no .clang-tidy config)
#   - Optional: -warnings-as-errors=*
#   - Optional: -fix -fix-errors to apply fixes
#
# Usage: run-clang-tidy.sh <build_dir> [path] [options]
#   build_dir  - CMake build directory with compile_commands.json
#   path       - File or directory to check (default: aster/)
#   options    - --fix, --warnings-as-errors
#
# Env: CLANG_TIDY - path to clang-tidy binary (default: clang-tidy)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "Usage: $0 <build_dir> [path] [options]"
  echo ""
  echo "  build_dir  - CMake build directory containing compile_commands.json"
  echo "  path       - File or directory to check (default: aster source tree)"
  echo "  options    - --fix                 Apply auto-fixes"
  echo "               --warnings-as-errors   Treat all diagnostics as errors (IREE style)"
  echo ""
  echo "Examples:"
  echo "  $0 build                      # Check all aster sources"
  echo "  $0 build lib/Support          # Check specific directory"
  echo "  $0 build lib/Support/Graph.cpp # Check single file"
  echo "  $0 build --fix                # Apply fixes"
  echo "  $0 build --warnings-as-errors # Strict mode"
  echo ""
  echo "Env: CLANG_TIDY - path to clang-tidy (default: clang-tidy)"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

BUILD_DIR="$1"
shift

# Resolve build dir (may be relative to cwd or script)
if [[ -d "$BUILD_DIR" ]]; then
  BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
elif [[ -d "${ASTER_ROOT}/${BUILD_DIR}" ]]; then
  BUILD_DIR="$(cd "${ASTER_ROOT}/${BUILD_DIR}" && pwd)"
elif [[ -d "${ASTER_ROOT}/../${BUILD_DIR}" ]]; then
  BUILD_DIR="$(cd "${ASTER_ROOT}/../${BUILD_DIR}" && pwd)"
else
  echo "Error: Build directory '$BUILD_DIR' not found"
  exit 1
fi

COMPILE_DB="${BUILD_DIR}/compile_commands.json"
if [[ ! -f "$COMPILE_DB" ]]; then
  echo "Error: compile_commands.json not found in $BUILD_DIR"
  echo "Configure with: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..."
  exit 1
fi

# Default path: aster source tree
PATH_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      EXTRA_ARGS+=( -fix -fix-errors )
      shift
      ;;
    --warnings-as-errors)
      EXTRA_ARGS+=( -warnings-as-errors='*' )
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      PATH_ARG="$1"
      shift
      break
      ;;
  esac
done

# Any remaining args go to clang-tidy
EXTRA_ARGS+=( "$@" )

# clang-tidy binary
if [[ -n "${CLANG_TIDY:-}" ]]; then
  CLANG_TIDY="$(realpath "$CLANG_TIDY")"
  if [[ ! -x "$CLANG_TIDY" ]]; then
    echo "Error: CLANG_TIDY='$CLANG_TIDY' is not executable"
    exit 1
  fi
else
  CLANG_TIDY="clang-tidy"
  if ! command -v "$CLANG_TIDY" &>/dev/null; then
    echo "Error: clang-tidy not found. Install it or set CLANG_TIDY."
    exit 1
  fi
fi

echo "Using: $CLANG_TIDY"
echo "Build dir: $BUILD_DIR"
echo "Path: $PATH_ARG"
echo ""

# Resolve path: default to aster source tree
if [[ -z "$PATH_ARG" ]]; then
  TARGET="$ASTER_ROOT"
elif [[ -d "$PATH_ARG" ]]; then
  TARGET="$(cd "$PATH_ARG" && pwd)"
elif [[ -f "$PATH_ARG" ]]; then
  TARGET="$(cd "$(dirname "$PATH_ARG")" && pwd)/$(basename "$PATH_ARG")"
elif [[ -d "${ASTER_ROOT}/${PATH_ARG}" ]]; then
  TARGET="$(cd "${ASTER_ROOT}/${PATH_ARG}" && pwd)"
elif [[ -f "${ASTER_ROOT}/${PATH_ARG}" ]]; then
  TARGET="${ASTER_ROOT}/${PATH_ARG}"
else
  echo "Error: Path '$PATH_ARG' not found"
  exit 1
fi

# Collect .cpp files
FILES=()
if [[ -f "$TARGET" ]]; then
  [[ "$TARGET" == *.cpp ]] && FILES=( "$TARGET" )
elif [[ -d "$TARGET" ]]; then
  while IFS= read -r -d '' f; do
    FILES+=( "$f" )
  done < <(find "$TARGET" -name "*.cpp" -print0 | sort -z)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .cpp files found under $TARGET"
  exit 0
fi

echo "Checking ${#FILES[@]} file(s)..."
echo ""

FAILED=0
for f in "${FILES[@]}"; do
  echo "=== $f ==="
  if ! "$CLANG_TIDY" -p "$BUILD_DIR" "$f" "${EXTRA_ARGS[@]}"; then
    FAILED=1
  fi
  echo ""
done

exit $FAILED
