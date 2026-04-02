#!/usr/bin/env bash
#
# tools/setup.sh - One-stop build script for ASTER
#
# Automates the full setup: prerequisites, shared LLVM, venv, cmake, build.
# Safe to re-run (idempotent). Works on macOS and Linux.

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

print_help() {
    echo "Usage: tools/setup.sh [OPTIONS]"
    echo "Examples:"
    echo "bash tools/setup.sh --with-hip --test-rocm --clang++=clang++-20"
    echo ""
    echo "One-stop build script for ASTER. Handles LLVM, venv, cmake, and build."
    echo ""
    echo "Options:"
    echo "  --llvm-only        Only set up shared LLVM (skip ASTER build)"
    echo "  --skip-llvm        Skip LLVM verification (assume shared LLVM is correct)"
    echo "  --skip-requirements  Skip Python requirements installation"
    echo "  --with-hip         Install ROCm SDK and build with HIP support (default on Linux)"
    echo "  --without-hip      Skip ROCm SDK, cross-compile mode only (default on macOS)"
    echo "  --rocm-target=T    Select ROCm target non-interactively (e.g. gfx94X)"
    echo "  --test-rocm        Test ROCm SDK after initialization (default: skip test)"
    echo "  --clang=PATH       Specify clang compiler    [default: clang]"
    echo "  --clang++=PATH     Specify clang++ compiler  [default: clang++]"
    echo "  --lld=PATH         Specify lld linker        [default: lld]"
    echo "  --python=PATH      Python interpreter to use when creating the environment"
    echo "  --venv=PATH        Use or create a specific Python environment"
    echo "  --venv-prompt=NAME Override the shell prompt shown inside the environment"
    echo "  --help             Show this help"
    echo ""
    echo "Environment variables (override defaults):"
    echo "  LLVM_INSTALL  Shared LLVM install prefix  [default: \$HOME/shared-llvm]"
    echo "  LLVM_BUILD    LLVM build directory         [default: \$HOME/llvm-build]"
    echo "  LLVM_PROJECT  LLVM source checkout         [default: \$HOME/llvm-project]"
}

# ---------------------------------------------------------------------------
# Common helpers
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
# Configuration
# ---------------------------------------------------------------------------

# Configurable environment variables (with defaults)
LLVM_INSTALL="${LLVM_INSTALL:-$HOME/shared-llvm}"
LLVM_BUILD="${LLVM_BUILD:-$HOME/llvm-build}"
LLVM_PROJECT="${LLVM_PROJECT:-$HOME/llvm-project}"

# Option variables (may be overridden by command-line arguments)
SKIP_LLVM=false
SKIP_REQUIREMENTS=false
LLVM_ONLY=false
HIP_EXPLICIT=""
ROCM_TARGET_EXPLICIT=""
SKIP_ROCM_TEST=true
CLANG_CMD="clang"
CLANGXX_CMD="clang++"
LLD_CMD="lld"
VENV_EXPLICIT=""
VENV_PROMPT_EXPLICIT=""
PYTHON_EXPLICIT=""

parse_arguments() {
    for arg in "$@"; do
        case "$arg" in
            --skip-llvm)         SKIP_LLVM=true ;;
            --skip-requirements) SKIP_REQUIREMENTS=true ;;
            --llvm-only)       LLVM_ONLY=true ;;
            --with-hip)        HIP_EXPLICIT=true ;;
            --without-hip)     HIP_EXPLICIT=false ;;
            --rocm-target=*)   ROCM_TARGET_EXPLICIT="${arg#*=}" ;;
            --test-rocm)       SKIP_ROCM_TEST=false ;;
            --clang=*)         CLANG_CMD="${arg#*=}" ;;
            --clang++=*)       CLANGXX_CMD="${arg#*=}" ;;
            --lld=*)           LLD_CMD="${arg#*=}" ;;
            --python=*)        PYTHON_EXPLICIT="${arg#*=}" ;;
            --venv=*)          VENV_EXPLICIT="${arg#*=}" ;;
            --venv-prompt=*)   VENV_PROMPT_EXPLICIT="${arg#*=}" ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                err "Unknown option: $arg"
                echo "Run 'tools/setup.sh --help' for usage."
                exit 1
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

# Script must be run from the ASTER repo root
ASTER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASTER_BUILD_DIR="${ASTER_DIR}/build"

resolve_virtual_env() {
    # Preserve any environment already active in the calling shell.
    local shell_virtual_env="${VIRTUAL_ENV:-}"

    VENV_PROMPT="aster"
    if [ -n "$VENV_EXPLICIT" ]; then
        VIRTUAL_ENV="$VENV_EXPLICIT"
    elif [ -n "$shell_virtual_env" ]; then
        VIRTUAL_ENV="$shell_virtual_env"
    else
        VIRTUAL_ENV="$ASTER_DIR/.aster"
    fi
    [ -n "$VENV_PROMPT_EXPLICIT" ] && VENV_PROMPT="$VENV_PROMPT_EXPLICIT"
}

resolve_with_hip() {
    if [ "$HIP_EXPLICIT" = "true" ]; then
        WITH_HIP=true
    elif [ "$HIP_EXPLICIT" = "false" ]; then
        WITH_HIP=false
    elif [ "$(uname)" = "Linux" ]; then
        WITH_HIP=true
    else
        WITH_HIP=false
    fi
}

phase1_detect_platform() {
    if [ "$(uname)" = "Darwin" ]; then
        PLATFORM="macos"
    elif command -v apt-get >/dev/null 2>&1; then
        PLATFORM="debian"
    elif command -v dnf >/dev/null 2>&1; then
        PLATFORM="fedora"
    else
        PLATFORM="unknown"
    fi
}

phase1_check_commands() {
    check_required_cmd git
    check_required_cmd cmake
    check_required_cmd ninja
    check_required_cmd "$CLANG_CMD"
    check_required_cmd "$CLANGXX_CMD"
    check_optional_cmd "$LLD_CMD"
    check_required_cmd uv
    check_required_cmd ccache
}

phase1_resolve_python() {
    if [ -n "$PYTHON_EXPLICIT" ]; then
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

phase1_prerequisites() {
    info "Phase 1: Checking prerequisites"
    MISSING=()
    phase1_detect_platform
    phase1_check_commands
    phase1_resolve_python

    if [ ${#MISSING[@]} -gt 0 ]; then
        err "Missing prerequisites: ${MISSING[*]}"
        exit 1
    fi
    echo ""
}

phase2_read_expected_commit() {
    LLVM_COMMIT_FILE="$ASTER_DIR/llvm/LLVM_COMMIT"
    if [ ! -f "$LLVM_COMMIT_FILE" ]; then
        err "Cannot find $LLVM_COMMIT_FILE"
        exit 1
    fi

    EXPECTED_COMMIT=$(head -1 "$LLVM_COMMIT_FILE" | tr -d '[:space:]')
    echo "  Expected LLVM commit: $EXPECTED_COMMIT"
}

phase2_check_installed_llvm() {
    LLVM_OK=false
    VCS_HEADER="$LLVM_INSTALL/include/llvm/Support/VCSRevision.h"
    if [ -f "$VCS_HEADER" ]; then
        INSTALLED_COMMIT=$(grep -o '[0-9a-f]\{40\}' "$VCS_HEADER" | head -1)
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
}

phase2_ensure_source_checkout() {
    if [ ! -d "$LLVM_PROJECT/.git" ]; then
        if ! ask "Clone llvm-project (shallow, ~500 MB)?"; then
            err "LLVM source is missing at $LLVM_PROJECT"
            exit 1
        fi
        echo "  Cloning llvm-project (shallow fetch of pinned commit)..."
        git init "$LLVM_PROJECT"
        git -C "$LLVM_PROJECT" remote add origin https://github.com/nicolasvasilache/llvm-project.git
        git -C "$LLVM_PROJECT" fetch --depth 1 origin "$EXPECTED_COMMIT"
        git -C "$LLVM_PROJECT" checkout FETCH_HEAD
    fi

    CURRENT_COMMIT=$(git -C "$LLVM_PROJECT" rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then
        echo "  Fetching pinned commit..."
        git -C "$LLVM_PROJECT" fetch --depth 1 origin "$EXPECTED_COMMIT"
        git -C "$LLVM_PROJECT" checkout FETCH_HEAD
    fi

    ok "LLVM source at correct commit"
}

phase2_build_shared_llvm_if_needed() {
    if [ "$LLVM_OK" = true ]; then
        return
    fi

    phase2_ensure_source_checkout
    echo ""
    echo "  Shared LLVM needs to be built. This takes 30-60+ minutes."
    echo "  Install prefix: $LLVM_INSTALL"
    echo "  Build dir:      $LLVM_BUILD"
    echo ""
    if ! ask "Build shared LLVM now?"; then
        err "Shared LLVM build was not confirmed"
        exit 1
    fi

    LLVM_LINKER_FLAGS=""
    if [ "$(uname)" = "Linux" ]; then
        if command -v "$LLD_CMD" >/dev/null 2>&1; then
            LLVM_LINKER_FLAGS="-DLLVM_USE_LINKER=${LLD_CMD}"
            ok "${LLD_CMD} found, using for faster link times"
        elif command -v ld.mold >/dev/null 2>&1; then
            LLVM_LINKER_FLAGS="-DLLVM_USE_LINKER=mold"
            ok "mold found, using for faster link times"
        fi
    fi

    export CC="$CLANG_CMD"
    export CXX="$CLANGXX_CMD"
    export LLVM_PROJECT="$LLVM_PROJECT"
    export LLVM_BUILD="$LLVM_BUILD"
    export LLVM_INSTALL="$LLVM_INSTALL"
    export LLVM_LINKER_FLAGS="$LLVM_LINKER_FLAGS"
    export LLVM_ENABLE_ASSERTIONS=ON
    bash "$ASTER_DIR/tools/build-llvm.sh"
    ok "Shared LLVM built and installed at $LLVM_INSTALL"
}

phase2_shared_llvm() {
    if [ "$SKIP_LLVM" = true ]; then
        info "Phase 2: Shared LLVM (skipped via --skip-llvm)"
        echo ""
        return
    fi

    info "Phase 2: Shared LLVM"
    phase2_read_expected_commit
    phase2_check_installed_llvm
    phase2_build_shared_llvm_if_needed
    echo ""
}

phase3_create_or_reuse_venv() {
    if [ -f "$VIRTUAL_ENV/bin/python" ]; then
        ok "venv exists at $VIRTUAL_ENV"
        return
    fi

    echo "  Creating venv at $VIRTUAL_ENV with $PYTHON..."
    if ! uv venv "$VIRTUAL_ENV" --seed --python "$PYTHON" --prompt "$VENV_PROMPT"; then
        err "Failed to create Python venv"
        exit 1
    fi
    ok "venv created"
}

phase3_verify_venv() {
    if ! "$VIRTUAL_ENV/bin/python" -c "import sys" 2>/dev/null; then
        err "venv python is broken at $VIRTUAL_ENV/bin/python"
        exit 1
    fi
}

phase3_install_requirements() {
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "requirements installation skipped (--skip-requirements)"
        return
    fi
    REQ_STAMP="$VIRTUAL_ENV/.requirements-stamp"
    if [ -f "$REQ_STAMP" ] && [ "$REQ_STAMP" -nt "$ASTER_DIR/requirements.txt" ]; then
        ok "requirements up to date"
        return
    fi

    echo "  Installing requirements..."
    if uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$ASTER_DIR/requirements.txt" 2>&1; then
        touch "$REQ_STAMP"
        ok "requirements installed"
    else
        err "Failed to install Python requirements"
        exit 1
    fi
}

phase3_select_rocm_target() {
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "requirements installation skipped (--skip-requirements)"
        return
    fi
    ROCM_REQ_FILES=()
    for f in "$ASTER_DIR"/requirements-amd-*.txt; do
        [ -f "$f" ] && ROCM_REQ_FILES+=("$f")
    done

    if [ ${#ROCM_REQ_FILES[@]} -eq 0 ]; then
        err "No requirements-amd-*.txt files found in $ASTER_DIR"
        exit 1
    fi

    if [ -n "$ROCM_TARGET_EXPLICIT" ]; then
        ROCM_REQ="$ASTER_DIR/requirements-amd-$ROCM_TARGET_EXPLICIT.txt"
        if [ ! -f "$ROCM_REQ" ]; then
            err "Unknown ROCm target: $ROCM_TARGET_EXPLICIT"
            exit 1
        fi
        ROCM_TARGET="$ROCM_TARGET_EXPLICIT"
        return
    fi

    if [ ${#ROCM_REQ_FILES[@]} -eq 1 ]; then
        ROCM_REQ="${ROCM_REQ_FILES[0]}"
        ROCM_TARGET=$(basename "$ROCM_REQ" .txt)
        ROCM_TARGET=${ROCM_TARGET#requirements-amd-}
        ok "Using only available ROCm target: $ROCM_TARGET"
        return
    fi

    if [ ! -t 0 ]; then
        err "Cannot prompt for ROCm target in non-interactive mode"
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

    if ! [[ "$ROCM_CHOICE" =~ ^[0-9]+$ ]] || [ "$ROCM_CHOICE" -lt 1 ] || [ "$ROCM_CHOICE" -gt ${#ROCM_REQ_FILES[@]} ]; then
        err "Invalid choice: $ROCM_CHOICE"
        exit 1
    fi

    ROCM_REQ="${ROCM_REQ_FILES[$((ROCM_CHOICE-1))]}"
    ROCM_TARGET=$(basename "$ROCM_REQ" .txt)
    ROCM_TARGET=${ROCM_TARGET#requirements-amd-}
}

phase3_install_rocm_sdk() {
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "ROCm SDK installation skipped (--skip-requirements)"
        return
    fi
    info "Installing ROCm SDK for $ROCM_TARGET"

    ROCM_STAMP="$VIRTUAL_ENV/.rocm-stamp-$ROCM_TARGET"
    if [ -f "$ROCM_STAMP" ] && [ "$ROCM_STAMP" -nt "$ROCM_REQ" ]; then
        ok "ROCm SDK ($ROCM_TARGET) already installed"
        return
    fi

    echo "  Installing ROCm SDK from $(head -1 "$ROCM_REQ" | sed 's/-i //')..."
    echo "  This downloads ~2 GB of AMD GPU libraries."
    echo ""
    if uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$ROCM_REQ" 2>&1; then
        rm -f "$VIRTUAL_ENV"/.rocm-stamp-* 2>/dev/null
        touch "$ROCM_STAMP"
        ok "ROCm SDK ($ROCM_TARGET) installed"
    else
        err "Failed to install ROCm SDK"
        exit 1
    fi
}

phase3_configure_rocm_env() {
    ROCM_DEVEL=$("$VIRTUAL_ENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel
    export ROCM_PATH="$ROCM_DEVEL"
    export HIP_PATH="$ROCM_DEVEL"
    CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/rocm' | tr '\n' ':' | sed 's/:$//')
    export PATH="$ROCM_DEVEL/bin:$CLEAN_PATH"
    ok "Isolated from system ROCm (ROCM_PATH=$ROCM_DEVEL)"
}

phase3_init_and_test_rocm() {
    echo "  Initializing ROCm SDK..."
    if ! "$VIRTUAL_ENV/bin/rocm-sdk" init 2>&1; then
        err "rocm-sdk init failed"
        exit 1
    fi
    ok "rocm-sdk initialized"

    if [ "$SKIP_ROCM_TEST" = false ]; then
        echo "  Testing ROCm SDK..."
        if ! "$VIRTUAL_ENV/bin/rocm-sdk" test 2>&1; then
            err "rocm-sdk test failed"
            exit 1
        fi
        ok "rocm-sdk test passed"
    else
        ok "rocm-sdk test skipped (use --test-rocm to enable)"
    fi

    if ! "$VIRTUAL_ENV/bin/rocm-sdk" path --cmake >/dev/null 2>&1; then
        err "rocm-sdk installed but 'rocm-sdk path --cmake' failed"
        exit 1
    fi
    ROCM_CMAKE_PREFIX=$("$VIRTUAL_ENV/bin/rocm-sdk" path --cmake 2>/dev/null)
    ok "rocm-sdk cmake prefix: $ROCM_CMAKE_PREFIX"
}

phase3_maybe_setup_rocm() {
    if [ "$WITH_HIP" != true ]; then
        return
    fi

    if [ "$(uname)" = "Darwin" ]; then
        err "--with-hip is only supported on Linux (AMD GPUs require Linux + ROCm)"
        exit 1
    fi

    phase3_select_rocm_target
    phase3_install_rocm_sdk
    phase3_configure_rocm_env
    phase3_init_and_test_rocm
    echo ""
}

phase3_update_activate_script() {
    ACTIVATE="$VIRTUAL_ENV/bin/activate"
    # Regenerate if the block is missing or doesn't include python_packages.
    if grep -q "python_packages" "$ACTIVATE" 2>/dev/null; then
        ok "activate script already configured"
        return
    fi

    # Strip any previous ASTER block before rewriting.
    if grep -q "ASTER setup (added by tools/setup.sh)" "$ACTIVATE" 2>/dev/null; then
        TMP=$(mktemp)
        sed '/# --- ASTER setup/,/# --- end ASTER setup ---/d' "$ACTIVATE" > "$TMP"
        mv "$TMP" "$ACTIVATE"
    fi

    echo "  Adding environment variables to activate script..."
    # LLVM_INSTALL is expanded now (at setup time) so the activate script is
    # pinned to the same install that was used to build ASTER.
    cat >> "$ACTIVATE" << 'ACTIVATE_EOF'

# --- ASTER setup (added by tools/setup.sh) ---
ACTIVATE_EOF
    printf 'export LLVM_INSTALL=%s\n' "$LLVM_INSTALL" >> "$ACTIVATE"
    printf 'export ASTER_SRC_DIR=%s\n' "$ASTER_DIR" >> "$ACTIVATE"
    cat >> "$ACTIVATE" << 'ACTIVATE_EOF'
export VENV_PURELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PATH=${LLVM_INSTALL}/bin:${VIRTUAL_ENV}/bin:${VENV_PURELIB}/_rocm_sdk_devel/bin:${PATH}
export PYTHONPATH=${VIRTUAL_ENV}/python_packages:${VENV_PURELIB}:${PYTHONPATH}
export LD_LIBRARY_PATH=${VENV_PURELIB}/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF

    ok "activate script updated"
}

phase3_generate_sandbox_activate() {
    local sandbox_dir="$ASTER_DIR/sandbox"
    local sandbox_bin="$sandbox_dir/bin"
    local sandbox_activate="$sandbox_bin/activate_sandbox"
    mkdir -p "$sandbox_bin"

    # Remove legacy scripts if present.
    rm -f "$sandbox_dir/activate.sh" "$sandbox_dir/deactivate.sh"

    cat > "$sandbox_activate" << SANDBOX_EOF
#!/usr/bin/env bash
#
# sandbox/bin/activate_sandbox - Activate the ASTER venv with sandbox paths.
#
# Usage:
#   source sandbox/bin/activate_sandbox
#
# To undo:
#   deactivate_sandbox

if [ -n "\${ASTER_SANDBOX_ACTIVE:-}" ]; then
    echo "sandbox already active (run deactivate_sandbox first)" >&2
    return 0
fi

deactivate_sandbox() {
    if [ -z "\${ASTER_SANDBOX_ACTIVE:-}" ]; then
        echo "sandbox is not active" >&2
        return 0
    fi

    if [ -n "\${_ASTER_OLD_PYTHONPATH+set}" ]; then
        if [ -n "\${_ASTER_OLD_PYTHONPATH}" ]; then
            export PYTHONPATH="\${_ASTER_OLD_PYTHONPATH}"
        else
            unset PYTHONPATH
        fi
    fi
    if [ -n "\${_ASTER_OLD_PATH+set}" ]; then
        if [ -n "\${_ASTER_OLD_PATH}" ]; then
            export PATH="\${_ASTER_OLD_PATH}"
        else
            unset PATH
        fi
    fi
    unset _ASTER_OLD_PYTHONPATH _ASTER_OLD_PATH ASTER_SANDBOX_ACTIVE
    unset -f deactivate_sandbox
    deactivate 2>/dev/null || true
}

# Save current PYTHONPATH and PATH so deactivate_sandbox can restore them.
export _ASTER_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export _ASTER_OLD_PATH="\${PATH:-}"

# shellcheck source=/dev/null
source "${VIRTUAL_ENV}/bin/activate"

# Prepend build-tree package directories and sandbox/bin.
export PYTHONPATH="${ASTER_BUILD_DIR}/python_packages\${PYTHONPATH:+:\${PYTHONPATH}}"
export PATH="${sandbox_bin}:${ASTER_BUILD_DIR}/bin\${PATH:+:\${PATH}}"

export ASTER_SANDBOX_ACTIVE=1
SANDBOX_EOF

    ok "sandbox/bin/activate_sandbox generated"
}


phase3_python_venv() {
    info "Phase 3: Python virtual environment"
    phase3_create_or_reuse_venv
    phase3_verify_venv
    phase3_install_requirements
    phase3_maybe_setup_rocm
    phase3_update_activate_script
    phase3_generate_sandbox_activate
    echo ""
}

phase4_detect_hip_support() {
    CMAKE_EXTRA_FLAGS=""
    CMAKE_PREFIX_CHAIN="$LLVM_INSTALL"

    if [ "$WITH_HIP" = true ]; then
        HIP_PREFIX="$ROCM_CMAKE_PREFIX/hip"
        if [ -d "$HIP_PREFIX" ]; then
            CMAKE_PREFIX_CHAIN="$CMAKE_PREFIX_CHAIN:$HIP_PREFIX"
            CMAKE_EXTRA_FLAGS="-DHIP_PLATFORM=amd"
            ok "HIP support enabled (from venv ROCm SDK)"
        else
            err "ROCm SDK installed but HIP cmake dir not found at $HIP_PREFIX"
            exit 1
        fi
        return
    fi

    if "$VIRTUAL_ENV/bin/rocm-sdk" path --cmake >/dev/null 2>&1; then
        HIP_PREFIX=$("$VIRTUAL_ENV/bin/rocm-sdk" path --cmake 2>/dev/null)/hip
        if [ -d "$HIP_PREFIX" ]; then
            CMAKE_PREFIX_CHAIN="$CMAKE_PREFIX_CHAIN:$HIP_PREFIX"
            CMAKE_EXTRA_FLAGS="-DHIP_PLATFORM=amd"
            ok "ROCm SDK detected in venv, enabling HIP support"
        fi
    else
        ok "No ROCm SDK (cross-compile mode, no GPU execution)"
    fi
}

phase4_needs_reconfigure() {
    NEED_RECONFIGURE=false
    if [ ! -f "$ASTER_BUILD_DIR/CMakeCache.txt" ] || [ ! -f "$ASTER_BUILD_DIR/build.ninja" ]; then
        NEED_RECONFIGURE=true
    elif [ "$WITH_HIP" = true ] && ! grep -q "HIP_PLATFORM" "$ASTER_BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        warn "Existing build lacks HIP support, reconfiguring for --with-hip"
        NEED_RECONFIGURE=true
    fi
}

phase4_select_linker() {
    ASTER_LINKER_FLAGS=""
    if [ "$(uname)" = "Linux" ]; then
        if command -v "$LLD_CMD" >/dev/null 2>&1; then
            ASTER_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=${LLD_CMD} -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=${LLD_CMD} -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=${LLD_CMD}"
            ok "Using ${LLD_CMD} for ASTER link"
        elif command -v ld.mold >/dev/null 2>&1; then
            ASTER_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=mold -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=mold"
            ok "Using mold for ASTER link"
        fi
    fi
}

phase4_configure_cmake() {
    phase4_select_linker

    echo "  Configuring cmake..."
    if [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
        CMAKE_PREFIX_CHAIN="$CMAKE_PREFIX_CHAIN:$CMAKE_PREFIX_PATH"
    fi

    if CMAKE_PREFIX_PATH="$CMAKE_PREFIX_CHAIN" "$VIRTUAL_ENV/bin/cmake" \
        -S "$ASTER_DIR" -B "$ASTER_BUILD_DIR" -GNinja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER="$CLANG_CMD" \
        -DCMAKE_CXX_COMPILER="$CLANGXX_CMD" \
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
        exit 1
    fi
}

phase4_cmake_configure() {
    info "Phase 4: CMake configure"
    mkdir -p "$ASTER_BUILD_DIR"

    phase4_detect_hip_support
    phase4_needs_reconfigure

    if [ "$NEED_RECONFIGURE" = false ]; then
        ok "cmake already configured (build/CMakeCache.txt exists)"
        echo "     To force reconfigure: rm $ASTER_BUILD_DIR/CMakeCache.txt && re-run"
        echo ""
        return
    fi

    phase4_configure_cmake
    echo ""
}

phase5_build() {
    info "Phase 5: Build"
    echo "  Running ninja install..."
    if "$VIRTUAL_ENV/bin/ninja" -C "$ASTER_BUILD_DIR" install; then
        ok "ASTER built"
    else
        err "Build failed"
        exit 1
    fi
    echo ""
}

print_summary() {
    info "Setup complete!"
    echo ""
    echo "  LLVM:    $LLVM_INSTALL"
    echo "  venv:    $VIRTUAL_ENV"
    echo "  build:   $ASTER_BUILD_DIR"
    echo ""
    echo "  Activate the venv:  source $VIRTUAL_ENV/bin/activate"
    echo "  Run lit tests:      $VIRTUAL_ENV/bin/lit $ASTER_BUILD_DIR/test -v"
    echo "  Run pytests:        cd $ASTER_DIR && $VIRTUAL_ENV/bin/pytest -n 16 ./test ./mlir_kernels ./contrib ./python"
    echo "  Rebuild:            ninja -C $ASTER_BUILD_DIR install"
}

main() {
    parse_arguments "$@"
    resolve_virtual_env
    resolve_with_hip

    phase1_prerequisites
    phase2_shared_llvm

    if [ "$LLVM_ONLY" = true ]; then
        info "Done (--llvm-only). Shared LLVM is ready at $LLVM_INSTALL"
        exit 0
    fi

    phase3_python_venv
    phase4_cmake_configure
    phase5_build
    print_summary
}

main "$@"
