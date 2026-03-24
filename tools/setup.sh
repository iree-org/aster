#!/usr/bin/env bash
#
# tools/setup.sh - One-stop build script for ASTER
#
# Automates the full setup: prerequisites, shared LLVM, venv, cmake, build.
# Safe to re-run (idempotent). Works on macOS and Linux.
#
# Usage:
#   tools/setup.sh              # Full setup and build
#   tools/setup.sh --llvm-only  # Only set up shared LLVM
#   tools/setup.sh --skip-llvm  # Skip LLVM check (assume it's correct)
#   tools/setup.sh --help       # Show usage
#
# Override paths via environment variables:
#   LLVM_INSTALL=$HOME/shared-llvm    # Where shared LLVM gets installed
#   LLVM_BUILD=$HOME/llvm-build       # LLVM build directory
#   LLVM_PROJECT=$HOME/llvm-project   # LLVM source checkout

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLVM_INSTALL="${LLVM_INSTALL:-$HOME/shared-llvm}"
LLVM_BUILD="${LLVM_BUILD:-$HOME/llvm-build}"
LLVM_PROJECT="${LLVM_PROJECT:-$HOME/llvm-project}"

# Script must be run from the ASTER repo root
ASTER_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Build type resolution (DEBUG=1 / RELEASE=1 / --debug / --release)
# ---------------------------------------------------------------------------
BUILD_MODE=""

# Detect worktree vs main repo
if [ -d "$ASTER_DIR/.aster-wt-"* ] 2>/dev/null; then
    WORKTREE_NAME="$(basename "$ASTER_DIR")"
    VIRTUAL_ENV="$ASTER_DIR/.aster-wt-$WORKTREE_NAME"
    VENV_PROMPT="aster-wt-$WORKTREE_NAME"
else
    VIRTUAL_ENV="$ASTER_DIR/.aster"
    VENV_PROMPT="aster"
fi

# ---------------------------------------------------------------------------
# Formatting
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
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_LLVM=false
LLVM_ONLY=false
HIP_EXPLICIT=""

for arg in "$@"; do
    case "$arg" in
        --skip-llvm)   SKIP_LLVM=true ;;
        --llvm-only)   LLVM_ONLY=true ;;
        --debug)       BUILD_MODE="debug" ;;
        --release)     BUILD_MODE="release" ;;
        --with-hip)    HIP_EXPLICIT=true ;;
        --without-hip) HIP_EXPLICIT=false ;;
        --help|-h)
            echo "Usage: tools/setup.sh [OPTIONS]"
            echo ""
            echo "One-stop build script for ASTER. Handles LLVM, venv, cmake, and build."
            echo ""
            echo "Options:"
            echo "  --debug         Build LLVM and ASTER in Debug mode (assertions ON)"
            echo "  --release       Build LLVM and ASTER in Release mode (fast, no debug info)"
            echo "                  Default (neither): RelWithDebInfo for ASTER, Release for LLVM"
            echo "                  Can also set via DEBUG=1 or RELEASE=1 environment variables"
            echo "  --llvm-only     Only set up shared LLVM (skip ASTER build)"
            echo "  --skip-llvm     Skip LLVM verification (assume shared LLVM is correct)"
            echo "  --with-hip      Install ROCm SDK and build with HIP support (default on Linux)"
            echo "  --without-hip   Skip ROCm SDK, cross-compile mode only (default on macOS)"
            echo "  --help          Show this help"
            echo ""
            echo "Environment variables (override defaults):"
            echo "  LLVM_INSTALL  Shared LLVM install prefix  [default: \$HOME/shared-llvm]"
            echo "  LLVM_BUILD    LLVM build directory         [default: \$HOME/llvm-build]"
            echo "  LLVM_PROJECT  LLVM source checkout         [default: \$HOME/llvm-project]"
            exit 0
            ;;
        *)
            err "Unknown option: $arg"
            echo "Run 'tools/setup.sh --help' for usage."
            exit 1
            ;;
    esac
done

# Resolve build type from CLI flags or env vars
if [ -z "$BUILD_MODE" ]; then
    if [ "${DEBUG:-}" = "1" ]; then
        BUILD_MODE="debug"
    elif [ "${RELEASE:-}" = "1" ]; then
        BUILD_MODE="release"
    fi
fi
BUILD_MODE="${BUILD_MODE:-default}"

case "$BUILD_MODE" in
    debug)
        ASTER_BUILD_TYPE="Debug"
        LLVM_BUILD_TYPE="Debug"
        BUILD_SUFFIX="-debug"
        ;;
    release)
        ASTER_BUILD_TYPE="Release"
        LLVM_BUILD_TYPE="Release"
        BUILD_SUFFIX="-release"
        ;;
    *)
        ASTER_BUILD_TYPE="RelWithDebInfo"
        LLVM_BUILD_TYPE="Release"
        BUILD_SUFFIX=""
        ;;
esac

# Suffix paths for non-default build types so variants coexist
LLVM_INSTALL="${LLVM_INSTALL}${BUILD_SUFFIX}"
LLVM_BUILD="${LLVM_BUILD}${BUILD_SUFFIX}"

info "Build type: $ASTER_BUILD_TYPE (ASTER), $LLVM_BUILD_TYPE (LLVM)"
echo ""

# Resolve WITH_HIP: default to true on Linux, false on macOS
if [ "$HIP_EXPLICIT" = "true" ]; then
    WITH_HIP=true
elif [ "$HIP_EXPLICIT" = "false" ]; then
    WITH_HIP=false
elif [ "$(uname)" = "Linux" ]; then
    WITH_HIP=true
else
    WITH_HIP=false
fi

# ---------------------------------------------------------------------------
# Phase 1: Prerequisites
# ---------------------------------------------------------------------------
info "Phase 1: Checking prerequisites"

MISSING=()

# Detect platform for install instructions
if [ "$(uname)" = "Darwin" ]; then
    PLATFORM="macos"
elif command -v apt-get >/dev/null 2>&1; then
    PLATFORM="debian"
elif command -v dnf >/dev/null 2>&1; then
    PLATFORM="fedora"
else
    PLATFORM="unknown"
fi

check_cmd() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd ($(command -v "$cmd"))"
    else
        err "$cmd not found"
        MISSING+=("$cmd")
    fi
}

check_cmd python3
check_cmd git
check_cmd cmake
check_cmd ninja
check_cmd clang
check_cmd clang++
if [ "$(uname)" = "Linux" ]; then
    check_cmd lld
fi
check_cmd uv
check_cmd ccache

# Resolve Python 3.12 via uv (preferred) or system python3
# uv manages its own python installs; we want 3.12 consistently everywhere.
PYTHON=""
if command -v uv >/dev/null 2>&1; then
    PYTHON=$(uv python find 3.12 2>/dev/null || true)
    if [ -n "$PYTHON" ]; then
        ok "python 3.12 via uv ($PYTHON)"
    fi
fi
if [ -z "$PYTHON" ] && command -v python3 >/dev/null 2>&1; then
    PYTHON=$(python3 -c "import sys; print(sys.executable)")
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
        ok "python3 $PY_VERSION ($PYTHON)"
    else
        err "python3 version $PY_VERSION is too old (need >= 3.12)"
        MISSING+=("python3>=3.12")
    fi
fi
if [ -z "$PYTHON" ]; then
    err "No suitable python found"
    MISSING+=("python3>=3.9")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    err "Missing prerequisites: ${MISSING[*]}"
    echo ""
    echo "To install everything at once:"
    echo ""
    case "$PLATFORM" in
        macos)
            echo "  brew install python3 git cmake ninja llvm uv ccache"
            echo ""
            echo "  If you don't have Homebrew: https://brew.sh"
            ;;
        debian)
            echo "  sudo apt-get update && sudo apt-get install -y git cmake ninja-build clang lld ccache"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  # Update PATH as printed by the uv installer, e.g.:"
            echo "  source \$HOME/.local/bin/env"
            echo "  uv python install 3.12"
            ;;
        fedora)
            echo "  sudo dnf install -y git cmake ninja-build clang lld ccache"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  # Update PATH as printed by the uv installer, e.g.:"
            echo "  source \$HOME/.local/bin/env"
            echo "  uv python install 3.12"
            ;;
        *)
            echo "  Install: python3 (>= 3.12), git, cmake, ninja, clang, lld, uv, ccache"
            echo "  uv: https://docs.astral.sh/uv/"
            ;;
    esac
    echo ""
    echo "Then re-run: tools/setup.sh"
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 2: Shared LLVM
# ---------------------------------------------------------------------------
if [ "$SKIP_LLVM" = true ]; then
    info "Phase 2: Shared LLVM (skipped via --skip-llvm)"
    echo ""
else
    info "Phase 2: Shared LLVM"

    # Read expected commit
    LLVM_COMMIT_FILE="$ASTER_DIR/llvm/LLVM_COMMIT"
    if [ ! -f "$LLVM_COMMIT_FILE" ]; then
        err "Cannot find $LLVM_COMMIT_FILE"
        echo "Are you running this from the ASTER repo root?"
        exit 1
    fi
    EXPECTED_COMMIT=$(head -1 "$LLVM_COMMIT_FILE" | tr -d '[:space:]')
    echo "  Expected LLVM commit: $EXPECTED_COMMIT"

    # Check if shared LLVM exists and has the right commit
    LLVM_OK=false
    VCS_HEADER="$LLVM_INSTALL/include/llvm/Support/VCSRevision.h"
    if [ -f "$VCS_HEADER" ]; then
        INSTALLED_COMMIT=$(grep 'LLVM_REVISION' "$VCS_HEADER" | sed 's/.*"\([0-9a-f]*\)".*/\1/')
        if [ "$INSTALLED_COMMIT" = "$EXPECTED_COMMIT" ]; then
            ok "Shared LLVM at $LLVM_INSTALL matches expected commit"
            LLVM_OK=true
        else
            warn "Shared LLVM commit mismatch"
            echo "     Installed: $INSTALLED_COMMIT"
            echo "     Expected:  $EXPECTED_COMMIT"
        fi
    else
        warn "No shared LLVM found at $LLVM_INSTALL"
    fi

    # Ensure LLVM build venv uses Python >= 3.12 (nanobind stubgen fails on <3.11)
    LLVM_VENV="$LLVM_BUILD/.venv"
    if [ -d "$LLVM_VENV" ]; then
        LLVM_VENV_PY_MINOR=$("$LLVM_VENV/bin/python" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$LLVM_VENV_PY_MINOR" -lt 12 ]; then
            warn "LLVM build venv uses Python 3.$LLVM_VENV_PY_MINOR (need >= 3.12), recreating..."
            rm -rf "$LLVM_VENV"
            if ! uv venv "$LLVM_VENV" --seed --python "$PYTHON"; then
                err "Failed to recreate LLVM build venv with Python >= 3.12"
                echo ""
                echo "Fix: uv python install 3.12"
                echo "Then re-run: tools/setup.sh"
                exit 1
            fi
            # Reinstall MLIR python requirements in the new venv
            MLIR_PYTHON_REQS="$LLVM_PROJECT/mlir/python/requirements.txt"
            if [ -f "$MLIR_PYTHON_REQS" ]; then
                echo "  Reinstalling MLIR python requirements in new venv..."
                uv pip install --python "$LLVM_VENV/bin/python" -r "$MLIR_PYTHON_REQS" 2>&1 \
                    || "$LLVM_VENV/bin/pip" install -r "$MLIR_PYTHON_REQS"
            fi
            # Force cmake reconfigure since Python changed
            LLVM_OK=false
            ok "LLVM build venv recreated with Python >= 3.12"
        fi
    fi

    if [ "$LLVM_OK" = false ]; then
        # Ensure LLVM source is cloned and at the right commit
        LLVM_SRC="$LLVM_PROJECT/llvm"
        if [ ! -d "$LLVM_PROJECT/.git" ]; then
            echo ""
            echo "  LLVM source not found at $LLVM_PROJECT"
            if ! ask "Clone llvm-project (shallow, ~500 MB)?"; then
                echo ""
                echo "To clone manually:"
                echo "  git init $LLVM_PROJECT"
                echo "  git -C $LLVM_PROJECT remote add origin https://github.com/nicolasvasilache/llvm-project.git"
                echo "  git -C $LLVM_PROJECT fetch --depth 1 origin $EXPECTED_COMMIT"
                echo "  git -C $LLVM_PROJECT checkout FETCH_HEAD"
                echo ""
                echo "Then re-run this script."
                exit 1
            fi
            echo "  Cloning llvm-project (shallow fetch of pinned commit)..."
            git init "$LLVM_PROJECT"
            git -C "$LLVM_PROJECT" remote add origin https://github.com/nicolasvasilache/llvm-project.git
            git -C "$LLVM_PROJECT" fetch --depth 1 origin "$EXPECTED_COMMIT"
            git -C "$LLVM_PROJECT" checkout FETCH_HEAD
        fi

        # Checkout the right commit (handles existing repo at wrong commit)
        CURRENT_COMMIT=$(git -C "$LLVM_PROJECT" rev-parse HEAD)
        if [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then
            echo "  Fetching pinned commit..."
            git -C "$LLVM_PROJECT" fetch --depth 1 origin "$EXPECTED_COMMIT"
            git -C "$LLVM_PROJECT" checkout FETCH_HEAD
        fi
        ok "LLVM source at correct commit"

        # Build shared LLVM
        echo ""
        echo "  Shared LLVM needs to be built. This takes 30-60+ minutes."
        echo "  Install prefix: $LLVM_INSTALL"
        echo "  Build dir:      $LLVM_BUILD"
        echo ""
        if ! ask "Build shared LLVM now?"; then
            echo ""
            echo "To build manually (see README_devs.md for full instructions):"
            echo "  mkdir -p $LLVM_BUILD && cd $LLVM_BUILD"
            echo "  cmake $LLVM_SRC -GNinja \\"
            echo "    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\"
            echo "    -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL \\"
            echo "    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \\"
            echo "    -DLLVM_ENABLE_PROJECTS='mlir;lld' \\"
            echo "    -DLLVM_TARGETS_TO_BUILD='AMDGPU' \\"
            echo "    -DLLVM_ENABLE_ASSERTIONS=ON \\"
            echo "    -DLLVM_USE_LINKER=lld \\"
            echo "    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \\"
            echo "    -DMLIR_ENABLE_EXECUTION_ENGINE=ON \\"
            echo "    -DMLIR_BUILD_MLIR_C_DYLIB=ON \\"
            echo "    -DLLVM_CCACHE_BUILD=ON"
            echo "  ninja install"
            echo "  ninja install FileCheck count not llvm-objdump"
            echo ""
            echo "Then re-run this script."
            exit 1
        fi

        # MLIR python bindings require a venv with pybind11/nanobind.
        # See: https://mlir.llvm.org/docs/Bindings/Python/#building
        LLVM_VENV="$LLVM_BUILD/.venv"
        if [ ! -d "$LLVM_VENV" ]; then
            echo "  Creating LLVM build venv with $PYTHON..."
            if ! uv venv "$LLVM_VENV" --seed --python "$PYTHON"; then
                err "Failed to create LLVM build venv at $LLVM_VENV"
                echo ""
                echo "Fix: uv python install 3.12"
                echo "Then re-run: tools/setup.sh"
                exit 1
            fi
            ok "LLVM build venv created"
        else
            ok "LLVM build venv exists at $LLVM_VENV"
        fi

        # Install MLIR python build requirements (pybind11, nanobind, etc.)
        MLIR_PYTHON_REQS="$LLVM_PROJECT/mlir/python/requirements.txt"
        if [ -f "$MLIR_PYTHON_REQS" ]; then
            echo "  Installing MLIR python requirements..."
            if ! uv pip install --python "$LLVM_VENV/bin/python" -r "$MLIR_PYTHON_REQS" 2>&1; then
                warn "uv failed, falling back to pip..."
                if ! "$LLVM_VENV/bin/pip" install -r "$MLIR_PYTHON_REQS"; then
                    err "Failed to install MLIR python requirements"
                    echo ""
                    echo "MLIR python bindings need packages from:"
                    echo "  $MLIR_PYTHON_REQS"
                    echo ""
                    echo "Try manually:"
                    echo "  source $LLVM_VENV/bin/activate"
                    echo "  pip install -r $MLIR_PYTHON_REQS"
                    echo "Then re-run: tools/setup.sh"
                    exit 1
                fi
            fi
            ok "MLIR python requirements installed"
        else
            warn "MLIR python requirements not found at $MLIR_PYTHON_REQS"
            echo "     MLIR python bindings may fail to build."
            echo "     Expected file: $MLIR_PYTHON_REQS"
        fi

        # Detect ccache
        CCACHE_FLAG=""
        if command -v ccache >/dev/null 2>&1; then
            CCACHE_FLAG="-DLLVM_CCACHE_BUILD=ON"
            ok "ccache found, will use for LLVM build"
        fi

        # Detect HIP/ROCm for LLVM build (optional)
        HIP_FLAGS=""
        if command -v rocm-sdk >/dev/null 2>&1; then
            HIP_PREFIX=$(rocm-sdk path --cmake 2>/dev/null)/hip
            if [ -d "$HIP_PREFIX" ]; then
                HIP_FLAGS="-DCMAKE_PREFIX_PATH=$HIP_PREFIX -DHIP_PLATFORM=amd"
                ok "ROCm SDK found, including HIP support in LLVM build"
            fi
        fi

        mkdir -p "$LLVM_BUILD"
        echo ""
        info "Building shared LLVM (this will take a while)..."

        # Point cmake at the venv python so it finds pybind11/nanobind
        # Use lld on Linux (required prereq), optional on macOS
        LINKER_FLAG=""
        if [ "$(uname)" = "Linux" ]; then
            LINKER_FLAG="-DLLVM_USE_LINKER=lld"
            ok "Using lld for LLVM link (Linux)"
        elif command -v ld.mold >/dev/null 2>&1; then
            LINKER_FLAG="-DLLVM_USE_LINKER=mold"
            ok "mold found, using for faster link times"
        fi

        # Only install what ASTER needs: MLIR/LLD/LLVM libraries + headers +
        # cmake configs + tblgen + python bindings. Skip 100+ unnecessary
        # binaries (mlir-opt, llc, opt, etc.) saving ~4 GB.
        if ! cmake -S "$LLVM_SRC" -B "$LLVM_BUILD" -GNinja \
            -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DLLVM_ENABLE_PROJECTS="mlir;lld" \
            -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
            -DMLIR_BUILD_MLIR_C_DYLIB=ON \
            -DLLVM_INSTALL_UTILS=ON \
            -DPython_EXECUTABLE="$LLVM_VENV/bin/python" \
            -DPython3_EXECUTABLE="$LLVM_VENV/bin/python" \
            $CCACHE_FLAG \
            $LINKER_FLAG \
            $HIP_FLAGS; then
            err "LLVM cmake configure failed"
            echo ""
            echo "Common causes:"
            echo "  - Missing python packages: source $LLVM_VENV/bin/activate && pip install -r $MLIR_PYTHON_REQS"
            echo "  - Missing system libraries: check cmake output above"
            echo "Build directory: $LLVM_BUILD"
            exit 1
        fi

        # Build and install only what ASTER needs:
        # - install-mlir-headers, install-mlir-libraries: MLIR headers + static libs
        # - install-lld-headers, install-lld-libraries: LLD headers + static libs
        # - install-llvm-headers, install-llvm-libraries: LLVM headers + static libs
        # - install-cmake-exports: cmake config files for find_package
        # - install-mlir-tblgen: needed for ASTER tablegen
        # - install-MLIR-C: MLIR C API dylib (python bindings)
        # - install-MLIRPythonModules: MLIR python bindings
        # This skips ~100 unnecessary binaries (mlir-opt, llc, opt, etc.)
        # saving ~4 GB in the install directory.
        if ! ninja -C "$LLVM_BUILD" \
            install-mlir-headers install-mlir-libraries \
            install-lld-headers install-lld-libraries \
            install-llvm-headers install-llvm-libraries \
            install-cmake-exports \
            install-mlir-tblgen \
            install-MLIR-C \
            install-MLIRPythonModules install-MLIRPythonSources \
            install-lld 2>&1; then
            # If targeted install fails, fall back to full install
            warn "Targeted install failed, falling back to full install..."
            if ! ninja -C "$LLVM_BUILD" install; then
                err "LLVM build failed"
                echo ""
                echo "Check the compiler errors above."
                echo "Build directory: $LLVM_BUILD"
                echo ""
                echo "To retry (without re-running cmake):"
                echo "  ninja -C $LLVM_BUILD install"
                exit 1
            fi
        fi

        # Install test tools (FileCheck, count, not, llvm-objdump)
        for tool in FileCheck count not llvm-objdump; do
            if [ -f "$LLVM_BUILD/bin/$tool" ]; then
                mkdir -p "$LLVM_INSTALL/bin"
                cp "$LLVM_BUILD/bin/$tool" "$LLVM_INSTALL/bin/$tool"
            fi
        done

        ok "Shared LLVM built and installed at $LLVM_INSTALL"
    fi
    echo ""
fi

# Exit early if --llvm-only
if [ "$LLVM_ONLY" = true ]; then
    info "Done (--llvm-only). Shared LLVM is ready at $LLVM_INSTALL"
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 3: Python venv
# ---------------------------------------------------------------------------
info "Phase 3: Python virtual environment"

if [ -f "$VIRTUAL_ENV/bin/python" ]; then
    ok "venv exists at $VIRTUAL_ENV"
else
    echo "  Creating venv at $VIRTUAL_ENV with $PYTHON..."
    if ! uv venv "$VIRTUAL_ENV" --seed --python "$PYTHON" --prompt "$VENV_PROMPT"; then
        err "Failed to create Python venv"
        echo ""
        echo "Common causes:"
        echo "  - Missing python 3.12: uv python install 3.12"
        echo "  - Broken uv installation"
        echo ""
        case "$PLATFORM" in
            debian)
                echo "Fix: sudo apt-get install -y python3-venv"
                ;;
            fedora)
                echo "Fix: sudo dnf install -y python3-devel"
                ;;
            macos)
                echo "Fix: brew reinstall python3"
                ;;
            *)
                echo "Fix: ensure python3 -m venv works"
                ;;
        esac
        echo "Then re-run: tools/setup.sh"
        exit 1
    fi
    ok "venv created"
fi

# Verify the venv python is functional
if ! "$VIRTUAL_ENV/bin/python" -c "import sys" 2>/dev/null; then
    err "venv python is broken at $VIRTUAL_ENV/bin/python"
    echo ""
    echo "Fix: remove the venv and re-run:"
    echo "  rm -rf $VIRTUAL_ENV"
    echo "  tools/setup.sh"
    exit 1
fi

# Install/update requirements (skip if unchanged since last install)
REQ_STAMP="$VIRTUAL_ENV/.requirements-stamp"
if [ -f "$REQ_STAMP" ] && [ "$REQ_STAMP" -nt "$ASTER_DIR/requirements.txt" ]; then
    ok "requirements up to date"
else
    echo "  Installing requirements..."
    if uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$ASTER_DIR/requirements.txt" 2>&1; then
        touch "$REQ_STAMP"
        ok "requirements installed"
    else
        err "Failed to install Python requirements"
        echo ""
        echo "Common causes:"
        echo "  - Network issue (pip needs to download packages)"
        echo "  - Incompatible Python version"
        echo "  - Missing system libraries for compiled packages"
        echo ""
        echo "Try manually:"
        echo "  uv pip install --python $VIRTUAL_ENV/bin/python -r $ASTER_DIR/requirements.txt"
        echo ""
        echo "If uv fails, try with pip directly:"
        echo "  $VIRTUAL_ENV/bin/pip install -r $ASTER_DIR/requirements.txt"
        exit 1
    fi
fi

# Install ROCm SDK if --with-hip
if [ "$WITH_HIP" = true ]; then
    if [ "$(uname)" = "Darwin" ]; then
        err "--with-hip is only supported on Linux (AMD GPUs require Linux + ROCm)"
        echo ""
        echo "On macOS, ASTER builds in cross-compile mode (no GPU execution)."
        echo "Use a Linux machine with AMD GPUs for --with-hip."
        exit 1
    fi

    # Find available ROCm requirements files
    ROCM_REQ_FILES=()
    for f in "$ASTER_DIR"/requirements-amd-*.txt; do
        [ -f "$f" ] && ROCM_REQ_FILES+=("$f")
    done

    if [ ${#ROCM_REQ_FILES[@]} -eq 0 ]; then
        err "No requirements-amd-*.txt files found in $ASTER_DIR"
        echo "Expected files like requirements-amd-gfx94X.txt"
        exit 1
    fi

    echo ""
    echo "  Available ROCm SDK targets:"
    for i in "${!ROCM_REQ_FILES[@]}"; do
        BASENAME=$(basename "${ROCM_REQ_FILES[$i]}" .txt)
        TARGET=${BASENAME#requirements-amd-}
        echo "    $((i+1))) $TARGET"
    done
    echo ""
    echo -n "  Which target? [1-${#ROCM_REQ_FILES[@]}] "
    read -r ROCM_CHOICE

    # Validate choice
    if ! [[ "$ROCM_CHOICE" =~ ^[0-9]+$ ]] || [ "$ROCM_CHOICE" -lt 1 ] || [ "$ROCM_CHOICE" -gt ${#ROCM_REQ_FILES[@]} ]; then
        err "Invalid choice: $ROCM_CHOICE"
        exit 1
    fi

    ROCM_REQ="${ROCM_REQ_FILES[$((ROCM_CHOICE-1))]}"
    ROCM_TARGET=$(basename "$ROCM_REQ" .txt)
    ROCM_TARGET=${ROCM_TARGET#requirements-amd-}
    info "Installing ROCm SDK for $ROCM_TARGET"

    ROCM_STAMP="$VIRTUAL_ENV/.rocm-stamp-$ROCM_TARGET"
    if [ -f "$ROCM_STAMP" ] && [ "$ROCM_STAMP" -nt "$ROCM_REQ" ]; then
        ok "ROCm SDK ($ROCM_TARGET) already installed"
    else
        echo "  Installing ROCm SDK from $(head -1 "$ROCM_REQ" | sed 's/-i //')..."
        echo "  This downloads ~2 GB of AMD GPU libraries."
        echo ""
        if uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$ROCM_REQ" 2>&1; then
            # Remove stamps for other targets (only one ROCm target at a time)
            rm -f "$VIRTUAL_ENV"/.rocm-stamp-* 2>/dev/null
            touch "$ROCM_STAMP"
            ok "ROCm SDK ($ROCM_TARGET) installed"
        else
            err "Failed to install ROCm SDK"
            echo ""
            echo "Common causes:"
            echo "  - Network issue (large download from AMD nightly index)"
            echo "  - Platform mismatch (ROCm SDK is Linux x86_64 only)"
            echo "  - Python version incompatibility"
            echo ""
            echo "Try manually:"
            echo "  uv pip install --python $VIRTUAL_ENV/bin/python -r $ROCM_REQ"
            exit 1
        fi
    fi

    # Isolate from any system ROCm at /opt/rocm*
    ROCM_DEVEL=$("$VIRTUAL_ENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel
    export ROCM_PATH="$ROCM_DEVEL"
    export HIP_PATH="$ROCM_DEVEL"
    # Strip /opt/rocm* from PATH so system hipconfig/rocm-smi are never found
    CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/rocm' | tr '\n' ':' | sed 's/:$//')
    export PATH="$ROCM_DEVEL/bin:$CLEAN_PATH"
    ok "Isolated from system ROCm (ROCM_PATH=$ROCM_DEVEL)"

    # Initialize ROCm SDK (unpacks libraries, sets up paths)
    echo "  Initializing ROCm SDK..."
    if ! "$VIRTUAL_ENV/bin/rocm-sdk" init 2>&1; then
        err "rocm-sdk init failed"
        echo ""
        echo "Common causes:"
        echo "  - Incomplete download (re-run to retry)"
        echo "  - Disk space (ROCm SDK needs several GB)"
        echo ""
        echo "Try manually:"
        echo "  $VIRTUAL_ENV/bin/rocm-sdk init"
        exit 1
    fi
    ok "rocm-sdk initialized"

    # Test ROCm SDK
    echo "  Testing ROCm SDK..."
    if ! "$VIRTUAL_ENV/bin/rocm-sdk" test 2>&1; then
        err "rocm-sdk test failed"
        echo ""
        echo "Common causes:"
        echo "  - No AMD GPU detected (check lspci or rocm-smi)"
        echo "  - Missing kernel driver (amdgpu)"
        echo "  - User not in 'render' or 'video' group"
        echo ""
        echo "Fix permissions:"
        echo "  sudo usermod -aG render,video \$USER"
        echo "  (log out and back in)"
        echo ""
        echo "Check GPU status:"
        echo "  $VIRTUAL_ENV/bin/rocm-sdk test"
        exit 1
    fi
    ok "rocm-sdk test passed"

    # Verify rocm-sdk path works (needed for cmake to find HIP)
    if ! "$VIRTUAL_ENV/bin/rocm-sdk" path --cmake >/dev/null 2>&1; then
        err "rocm-sdk installed but 'rocm-sdk path --cmake' failed"
        echo ""
        echo "The ROCm SDK may be corrupt. Try reinstalling:"
        echo "  uv pip install --python $VIRTUAL_ENV/bin/python --force-reinstall -r $ROCM_REQ"
        echo "  $VIRTUAL_ENV/bin/rocm-sdk init"
        exit 1
    fi
    ROCM_CMAKE_PREFIX=$("$VIRTUAL_ENV/bin/rocm-sdk" path --cmake 2>/dev/null)
    ok "rocm-sdk cmake prefix: $ROCM_CMAKE_PREFIX"
    echo ""
fi

# Inject env vars into activate script if not already present
ACTIVATE="$VIRTUAL_ENV/bin/activate"
if ! grep -q "CMAKE_PREFIX_PATH" "$ACTIVATE" 2>/dev/null; then
    echo "  Adding environment variables to activate script..."

    # Build the activate snippet based on venv type
    if [ "$VENV_PROMPT" = "aster" ]; then
        # Main repo venv
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'

# --- ASTER setup (added by tools/setup.sh) ---
export PATH=${PWD}/.aster/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/bin/:${PATH}
export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/lib
export LLVM_INSTALL=${HOME}/shared-llvm
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF
    else
        # Worktree venv
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'

# --- ASTER setup (added by tools/setup.sh) ---
export WORKTREE_NAME=$(basename $(pwd))
export PATH=${PWD}/.aster-wt-${WORKTREE_NAME}/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/bin/:${PATH}
export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster-wt-${WORKTREE_NAME}/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/lib
export LLVM_INSTALL=${HOME}/shared-llvm
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF
    fi
    ok "activate script updated"
else
    ok "activate script already configured"
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 4: CMake configure
# ---------------------------------------------------------------------------
info "Phase 4: CMake configure"

BUILD_DIR="$ASTER_DIR/build${BUILD_SUFFIX}"
mkdir -p "$BUILD_DIR"

# Detect platform and HIP support
CMAKE_EXTRA_FLAGS=""
if [ "$WITH_HIP" = true ]; then
    # --with-hip: use the rocm-sdk we installed in the venv
    HIP_PREFIX="$ROCM_CMAKE_PREFIX/hip"
    if [ -d "$HIP_PREFIX" ]; then
        CMAKE_EXTRA_FLAGS="-DCMAKE_PREFIX_PATH=$HIP_PREFIX -DHIP_PLATFORM=amd"
        ok "HIP support enabled (from venv ROCm SDK)"
    else
        err "ROCm SDK installed but HIP cmake dir not found at $HIP_PREFIX"
        echo ""
        echo "Expected directory: $HIP_PREFIX"
        echo "rocm-sdk cmake prefix: $ROCM_CMAKE_PREFIX"
        echo ""
        echo "Try reinstalling the ROCm SDK:"
        echo "  uv pip install --python $VIRTUAL_ENV/bin/python --force-reinstall -r $ROCM_REQ"
        exit 1
    fi
elif "$VIRTUAL_ENV/bin/rocm-sdk" path --cmake >/dev/null 2>&1; then
    # ROCm SDK already in venv from a previous --with-hip run
    HIP_PREFIX=$("$VIRTUAL_ENV/bin/rocm-sdk" path --cmake 2>/dev/null)/hip
    if [ -d "$HIP_PREFIX" ]; then
        CMAKE_EXTRA_FLAGS="-DCMAKE_PREFIX_PATH=$HIP_PREFIX -DHIP_PLATFORM=amd"
        ok "ROCm SDK detected in venv, enabling HIP support"
    fi
else
    ok "No ROCm SDK (cross-compile mode, no GPU execution)"
fi

NEED_RECONFIGURE=false
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
    NEED_RECONFIGURE=true
elif [ "$WITH_HIP" = true ] && ! grep -q "HIP_PLATFORM" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    warn "Existing build lacks HIP support, reconfiguring for --with-hip"
    NEED_RECONFIGURE=true
fi

if [ "$NEED_RECONFIGURE" = false ]; then
    ok "cmake already configured (build/CMakeCache.txt exists)"
    echo "     To force reconfigure: rm $BUILD_DIR/CMakeCache.txt && re-run"
else
    # Use lld on Linux (required prereq), optional on macOS
    ASTER_LINKER_FLAGS=""
    if [ "$(uname)" = "Linux" ]; then
        ASTER_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
        ok "Using lld for ASTER link (Linux)"
    elif command -v ld.mold >/dev/null 2>&1; then
        ASTER_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=mold"
        ok "Using mold for ASTER link"
    fi

    echo "  Configuring cmake..."
    if CMAKE_PREFIX_PATH="$LLVM_INSTALL" "$VIRTUAL_ENV/bin/cmake" \
        -S "$ASTER_DIR" -B "$BUILD_DIR" -GNinja \
        -DCMAKE_BUILD_TYPE="$ASTER_BUILD_TYPE" \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
        -DLLVM_EXTERNAL_LIT="$VIRTUAL_ENV/bin/lit" \
        -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
        -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
        -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=aster \
        $ASTER_LINKER_FLAGS \
        $CMAKE_EXTRA_FLAGS; then
        ok "cmake configured"
    else
        err "cmake configure failed"
        echo ""
        echo "Common fixes:"
        echo "  - LLVM commit mismatch: rebuild shared LLVM (tools/setup.sh --llvm-only)"
        echo "  - Python issues: check $VIRTUAL_ENV/bin/python exists"
        exit 1
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 5: Build
# ---------------------------------------------------------------------------
info "Phase 5: Build"

echo "  Running ninja install..."
if "$VIRTUAL_ENV/bin/ninja" -C "$BUILD_DIR" install; then
    ok "ASTER built and installed"
else
    err "Build failed"
    echo "Check the output above for errors."
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
info "Setup complete!"
echo ""
echo "  LLVM:    $LLVM_INSTALL"
echo "  venv:    $VIRTUAL_ENV"
echo "  build:   $BUILD_DIR"
echo ""
echo "  Activate the venv:  source $VIRTUAL_ENV/bin/activate"
echo "  Run lit tests:      $VIRTUAL_ENV/bin/lit $BUILD_DIR/test -v"
echo "  Run pytests:        cd $ASTER_DIR && $VIRTUAL_ENV/bin/pytest -n 16 ./test ./mlir_kernels ./contrib ./python"
echo "  Rebuild:            ninja -C $BUILD_DIR install"
