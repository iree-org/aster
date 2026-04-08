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
    echo "  --no-install       Only build (skip ninja install)"
    echo "  --help             Show this help"
    echo ""
    echo "Environment variables (override defaults):"
    echo "  LLVM_INSTALL      Shared LLVM install prefix  [default: \$HOME/shared-llvm]"
    echo "  LLVM_BUILD        LLVM build directory         [default: \$HOME/llvm-build]"
    echo "  LLVM_PROJECT      LLVM source checkout         [default: \$HOME/llvm-project]"
    echo "  ASTER_ENABLE_CPU  When set to 1, build shared LLVM with X86 target and"
    echo "                    configure aster with -DASTER_ENABLE_CPU=ON (contrib/cpu AMX)"
    echo ""
    echo "Examples:"
    echo "  ASTER_ENABLE_CPU=1 tools/setup.sh     # include x86 AMX contrib"
}

# ---------------------------------------------------------------------------
# Common helpers (shared with benchmarks/setup.sh)
# ---------------------------------------------------------------------------

# shellcheck source=tools/common.sh
source "$(dirname "$0")/common.sh"

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
# ASTER_ENABLE_CPU is an env var, not a CLI flag. Normalise to true/false.
case "${ASTER_ENABLE_CPU:-0}" in
    1|ON|on|true|TRUE|yes|YES) WITH_CPU=true ;;
    *) WITH_CPU=false ;;
esac
ROCM_TARGET_EXPLICIT=""
SKIP_ROCM_TEST=true
CLANG_CMD="clang"
CLANGXX_CMD="clang++"
LLD_CMD="lld"
VENV_EXPLICIT=""
VENV_PROMPT_EXPLICIT=""
PYTHON_EXPLICIT=""
NO_INSTALL=false

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
            --no-install)      NO_INSTALL=true ;;
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
    resolve_python    # from common.sh
}

phase1_map_package() {
    # Map a missing command to a package name for the current PLATFORM.
    # Prints nothing if the item is handled separately (e.g. uv, python).
    local cmd="$1"
    case "$cmd" in
        clang|clang++|clang-*|clang++-*) echo "clang" ;;
        lld|lld-*|ld.lld) echo "lld" ;;
        ninja)
            case "$PLATFORM" in
                debian|fedora) echo "ninja-build" ;;
                macos)        echo "ninja" ;;
            esac ;;
        cmake)   echo "cmake" ;;
        ccache)  echo "ccache" ;;
        git)     echo "git" ;;
        uv|python3*) ;; # handled by phase1_print_install_hint separately
        *) echo "$cmd" ;;
    esac
}

phase1_print_install_hint() {
    # Dedup packages using a sorted uniq over the mapped names.
    local pkgs=() needs_uv=false needs_python=false item mapped existing found
    for item in "${MISSING[@]}"; do
        case "$item" in
            uv) needs_uv=true; continue ;;
            python3*) needs_python=true; continue ;;
        esac
        mapped=$(phase1_map_package "$item")
        [ -z "$mapped" ] && continue
        found=false
        for existing in "${pkgs[@]}"; do
            [ "$existing" = "$mapped" ] && { found=true; break; }
        done
        [ "$found" = false ] && pkgs+=("$mapped")
    done

    echo ""
    echo "  To install the missing prerequisites, run:"
    case "$PLATFORM" in
        debian)
            if [ ${#pkgs[@]} -gt 0 ]; then
                echo "    sudo apt-get update && sudo apt-get install -y ${pkgs[*]}"
            fi
            [ "$needs_python" = true ] && \
                echo "    sudo apt-get install -y python3.12 python3.12-venv"
            ;;
        fedora)
            [ ${#pkgs[@]} -gt 0 ] && echo "    sudo dnf install -y ${pkgs[*]}"
            [ "$needs_python" = true ] && echo "    sudo dnf install -y python3.12"
            ;;
        macos)
            [ ${#pkgs[@]} -gt 0 ] && echo "    brew install ${pkgs[*]}"
            [ "$needs_python" = true ] && echo "    brew install python@3.12"
            ;;
        *)
            echo "    (unknown platform: install ${MISSING[*]} via your package manager)"
            ;;
    esac
    [ "$needs_uv" = true ] && \
        echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
}

phase1_prerequisites() {
    info "Phase 1: Checking prerequisites"
    MISSING=()
    phase1_detect_platform
    phase1_check_commands
    phase1_resolve_python

    if [ ${#MISSING[@]} -gt 0 ]; then
        err "Missing prerequisites: ${MISSING[*]}"
        phase1_print_install_hint
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

    # ASTER_ENABLE_CPU additionally requires an x86-capable llvm-mc. A shared
    # LLVM built with LLVM_TARGETS_TO_BUILD=AMDGPU only is not usable for the
    # contrib/cpu AMX lit tests -- force a rebuild in that case.
    if [ "$WITH_CPU" = true ] && [ "$LLVM_OK" = true ]; then
        if [ ! -x "$LLVM_INSTALL/bin/llvm-mc" ]; then
            warn "ASTER_ENABLE_CPU=1: shared LLVM has no llvm-mc binary, forcing rebuild"
            LLVM_OK=false
        elif ! "$LLVM_INSTALL/bin/llvm-mc" --version 2>/dev/null \
                | grep -qiE '(^|[^a-z])x86([^a-z]|$)'; then
            warn "ASTER_ENABLE_CPU=1: shared LLVM llvm-mc has no x86 target, forcing rebuild"
            LLVM_OK=false
        else
            ok "ASTER_ENABLE_CPU=1: shared LLVM llvm-mc has x86 target"
        fi
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
    if [ "$WITH_CPU" = true ]; then
        export LLVM_TARGETS_TO_BUILD="AMDGPU;X86"
        ok "ASTER_ENABLE_CPU=1: building LLVM with targets AMDGPU;X86"
    fi
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
    create_or_reuse_venv "$VIRTUAL_ENV" "$VENV_PROMPT"    # from common.sh
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
    install_requirements "$VIRTUAL_ENV" "$ASTER_DIR/requirements.txt"    # from common.sh
}

phase3_select_rocm_target() {
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "requirements installation skipped (--skip-requirements)"
        return
    fi
    select_rocm_target "$ASTER_DIR" "$ROCM_TARGET_EXPLICIT"    # from common.sh
}

phase3_install_rocm_sdk() {
    if [ "$SKIP_REQUIREMENTS" = true ]; then
        ok "ROCm SDK installation skipped (--skip-requirements)"
        return
    fi
    install_rocm_sdk "$VIRTUAL_ENV"    # from common.sh
}

phase3_configure_rocm_env() {
    configure_rocm_env "$VIRTUAL_ENV"    # from common.sh
}

phase3_init_and_test_rocm() {
    local do_test="false"
    [ "$SKIP_ROCM_TEST" = false ] && do_test="true"
    init_rocm_sdk "$VIRTUAL_ENV" "$do_test"    # from common.sh

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

    local cpu_flag=""
    if [ "$WITH_CPU" = true ]; then
        cpu_flag="-DASTER_ENABLE_CPU=ON"
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
        ${cpu_flag} \
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
    echo "  Running ninja..."
    NINJA_ARGS="install"
    if [ "$NO_INSTALL" = true ]; then
        NINJA_ARGS=""
    fi
    if "$VIRTUAL_ENV/bin/ninja" -C "$ASTER_BUILD_DIR" $NINJA_ARGS; then
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
