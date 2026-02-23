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

# Detect worktree vs main repo
if [ -d "$ASTER_DIR/.aster-wt-"* ] 2>/dev/null; then
    WORKTREE_NAME="$(basename "$ASTER_DIR")"
    VENV_DIR="$ASTER_DIR/.aster-wt-$WORKTREE_NAME"
    VENV_PROMPT="aster-wt-$WORKTREE_NAME"
else
    VENV_DIR="$ASTER_DIR/.aster"
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

for arg in "$@"; do
    case "$arg" in
        --skip-llvm) SKIP_LLVM=true ;;
        --llvm-only) LLVM_ONLY=true ;;
        --help|-h)
            echo "Usage: tools/setup.sh [OPTIONS]"
            echo ""
            echo "One-stop build script for ASTER. Handles LLVM, venv, cmake, and build."
            echo ""
            echo "Options:"
            echo "  --llvm-only   Only set up shared LLVM (skip ASTER build)"
            echo "  --skip-llvm   Skip LLVM verification (assume shared LLVM is correct)"
            echo "  --help        Show this help"
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

# ---------------------------------------------------------------------------
# Phase 1: Prerequisites
# ---------------------------------------------------------------------------
info "Phase 1: Checking prerequisites"

MISSING=()

check_cmd() {
    local cmd="$1" install_hint="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd ($(command -v "$cmd"))"
    else
        err "$cmd not found"
        echo "     Install: $install_hint"
        MISSING+=("$cmd")
    fi
}

check_cmd python3  "https://www.python.org/downloads/ or 'brew install python3'"
check_cmd git      "'brew install git' or 'apt install git'"
check_cmd clang    "'brew install llvm' or 'apt install clang'"
check_cmd clang++  "(installed with clang)"
check_cmd uv       "'brew install uv' or 'pip install uv' -- https://docs.astral.sh/uv/"

# Check python version >= 3.9
if command -v python3 >/dev/null 2>&1; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 9 ]; then
        ok "python3 version $PY_VERSION (>= 3.9)"
    else
        err "python3 version $PY_VERSION is too old (need >= 3.9)"
        MISSING+=("python3>=3.9")
    fi
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    err "Missing prerequisites: ${MISSING[*]}"
    echo "Install the above and re-run this script."
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

    if [ "$LLVM_OK" = false ]; then
        # Ensure LLVM source is cloned and at the right commit
        LLVM_SRC="$LLVM_PROJECT/llvm"
        if [ ! -d "$LLVM_PROJECT/.git" ]; then
            echo ""
            echo "  LLVM source not found at $LLVM_PROJECT"
            if ! ask "Clone llvm-project? (~2 GB download)"; then
                echo ""
                echo "To clone manually:"
                echo "  git clone https://github.com/llvm/llvm-project.git $LLVM_PROJECT"
                echo "  git -C $LLVM_PROJECT checkout $EXPECTED_COMMIT"
                echo ""
                echo "Then re-run this script."
                exit 1
            fi
            echo "  Cloning llvm-project..."
            git clone https://github.com/llvm/llvm-project.git "$LLVM_PROJECT"
        fi

        # Checkout the right commit
        CURRENT_COMMIT=$(git -C "$LLVM_PROJECT" rev-parse HEAD)
        if [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then
            echo "  Checking out pinned commit..."
            git -C "$LLVM_PROJECT" fetch origin "$EXPECTED_COMMIT" 2>/dev/null || \
                git -C "$LLVM_PROJECT" fetch origin
            git -C "$LLVM_PROJECT" checkout "$EXPECTED_COMMIT"
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

        # Create a temporary venv for LLVM's python bindings build
        LLVM_VENV="$LLVM_BUILD/.venv"
        if [ ! -d "$LLVM_VENV" ]; then
            echo "  Creating LLVM build venv..."
            uv venv "$LLVM_VENV" --seed -p 3.12 2>/dev/null || \
                python3 -m venv "$LLVM_VENV"
        fi

        # Install MLIR python build requirements
        if [ -f "$LLVM_SRC/mlir/python/requirements.txt" ]; then
            uv pip install --python "$LLVM_VENV/bin/python" \
                -r "$LLVM_SRC/mlir/python/requirements.txt" 2>/dev/null || \
            "$LLVM_VENV/bin/pip" install -r "$LLVM_SRC/mlir/python/requirements.txt"
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

        # Activate LLVM venv for python bindings
        # shellcheck disable=SC1091
        source "$LLVM_VENV/bin/activate"

        cmake -S "$LLVM_SRC" -B "$LLVM_BUILD" -GNinja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DLLVM_ENABLE_PROJECTS="mlir;lld" \
            -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
            -DMLIR_BUILD_MLIR_C_DYLIB=ON \
            $CCACHE_FLAG \
            $HIP_FLAGS

        if ! ninja -C "$LLVM_BUILD" install; then
            err "LLVM build failed"
            echo "Check the output above for errors."
            echo "Build directory: $LLVM_BUILD"
            exit 1
        fi

        # Install test tools
        ninja -C "$LLVM_BUILD" install FileCheck count not llvm-objdump 2>/dev/null || true

        # Some systems need manual copy of test tools
        for tool in FileCheck count not llvm-objdump; do
            if [ -f "$LLVM_BUILD/bin/$tool" ] && [ ! -f "$LLVM_INSTALL/bin/$tool" ]; then
                cp "$LLVM_BUILD/bin/$tool" "$LLVM_INSTALL/bin/$tool"
            fi
        done

        deactivate 2>/dev/null || true

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

if [ -f "$VENV_DIR/bin/python" ]; then
    ok "venv exists at $VENV_DIR"
else
    echo "  Creating venv at $VENV_DIR..."
    python3 -m venv --prompt "$VENV_PROMPT" "$VENV_DIR"
    ok "venv created"
fi

# Install/update requirements (skip if unchanged since last install)
REQ_STAMP="$VENV_DIR/.requirements-stamp"
if [ -f "$REQ_STAMP" ] && [ "$REQ_STAMP" -nt "$ASTER_DIR/requirements.txt" ]; then
    ok "requirements up to date"
else
    echo "  Installing requirements..."
    if uv pip install --python "$VENV_DIR/bin/python" -r "$ASTER_DIR/requirements.txt" >/dev/null 2>&1; then
        touch "$REQ_STAMP"
        ok "requirements installed"
    else
        err "Failed to install requirements"
        echo "Try manually: uv pip install --python $VENV_DIR/bin/python -r $ASTER_DIR/requirements.txt"
        exit 1
    fi
fi

# Inject env vars into activate script if not already present
ACTIVATE="$VENV_DIR/bin/activate"
if ! grep -q "CMAKE_PREFIX_PATH" "$ACTIVATE" 2>/dev/null; then
    echo "  Adding environment variables to activate script..."

    # Build the activate snippet based on venv type
    if [ "$VENV_PROMPT" = "aster" ]; then
        # Main repo venv
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'

# --- ASTER setup (added by tools/setup.sh) ---
export PATH=${PWD}/.aster/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):${PATH}
export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LLVM_INSTALL=${HOME}/shared-llvm
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF
    else
        # Worktree venv
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'

# --- ASTER setup (added by tools/setup.sh) ---
export WORKTREE_NAME=$(basename $(pwd))
export PATH=${PWD}/.aster-wt-${WORKTREE_NAME}/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):${PATH}
export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster-wt-${WORKTREE_NAME}/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
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

BUILD_DIR="$ASTER_DIR/build"
mkdir -p "$BUILD_DIR"

# Detect platform and HIP support
CMAKE_EXTRA_FLAGS=""
if command -v rocm-sdk >/dev/null 2>&1; then
    HIP_PREFIX=$(rocm-sdk path --cmake 2>/dev/null)/hip
    if [ -d "$HIP_PREFIX" ]; then
        CMAKE_EXTRA_FLAGS="-DCMAKE_PREFIX_PATH=\"$HIP_PREFIX\" -DHIP_PLATFORM=amd"
        ok "ROCm SDK detected, enabling HIP support"
    fi
else
    ok "No ROCm SDK (cross-compile mode, no GPU execution)"
fi

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    ok "cmake already configured (build/CMakeCache.txt exists)"
    echo "     To force reconfigure: rm $BUILD_DIR/CMakeCache.txt && re-run"
else
    echo "  Configuring cmake..."
    if CMAKE_PREFIX_PATH="$LLVM_INSTALL" cmake \
        -S "$ASTER_DIR" -B "$BUILD_DIR" -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INSTALL_PREFIX="$VENV_DIR" \
        -DLLVM_EXTERNAL_LIT="$VENV_DIR/bin/lit" \
        -DPython_EXECUTABLE="$VENV_DIR/bin/python" \
        -DPython3_EXECUTABLE="$VENV_DIR/bin/python" \
        $CMAKE_EXTRA_FLAGS; then
        ok "cmake configured"
    else
        err "cmake configure failed"
        echo ""
        echo "Common fixes:"
        echo "  - LLVM commit mismatch: rebuild shared LLVM (tools/setup.sh --llvm-only)"
        echo "  - Python issues: check $VENV_DIR/bin/python exists"
        exit 1
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 5: Build
# ---------------------------------------------------------------------------
info "Phase 5: Build"

echo "  Running ninja install..."
if ninja -C "$BUILD_DIR" install; then
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
echo "  venv:    $VENV_DIR"
echo "  build:   $BUILD_DIR"
echo ""
echo "  Activate the venv:  source $VENV_DIR/bin/activate"
echo "  Run lit tests:      $VENV_DIR/bin/lit $BUILD_DIR/test -v"
echo "  Run pytests:        cd $ASTER_DIR && $VENV_DIR/bin/pytest -n 16 ./test ./mlir_kernels ./contrib"
echo "  Rebuild:            ninja -C $BUILD_DIR install"
