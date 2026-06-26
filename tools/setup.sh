#!/usr/bin/env bash
#
# tools/setup.sh - Setup script for ASTER
#
# Configures prerequisites: shared LLVM, venv, ROCm, and cmake. By DEFAULT it
# does NOT build ASTER -- it prints the activate / build / run commands. Pass
# --build to also run the ASTER build (ninja install) in one step.
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
    echo "  --install-prerequisites  Install missing tools locally (no sudo) via uv"
    echo "  --with-hip         Install ROCm SDK and build with HIP support (default on Linux)"
    echo "  --without-hip      Skip ROCm SDK, cross-compile mode only (default on macOS)"
    echo "  --rocm-target=T    Select pip ROCm target (gfx94X, gfx950, gfx120X-all)"
    echo "                     gfx1250 has no pip ROCm SDK -- use --rocm-path instead"
    echo "  --rocm-path=DIR    Use a preinstalled ROCm at DIR (bypass uv pip install;"
    echo "                     expects DIR/lib/libamdhip64.so and DIR/lib/cmake/hip)"
    echo "  --test-rocm        Test ROCm SDK after initialization (default: skip test)"
    echo "  --clang=PATH       Specify clang compiler    [default: clang]"
    echo "  --clang++=PATH     Specify clang++ compiler  [default: clang++]"
    echo "  --ccache=PATH      Specify ccache binary     [default: ccache on PATH]"
    echo "  --lld=PATH         Specify lld linker        [default: lld]"
    echo "  --python=PATH      Python interpreter to use when creating the environment"
    echo "  --venv=PATH        Use or create a specific Python environment"
    echo "  --venv-prompt=NAME Override the shell prompt shown inside the environment"
    echo "  --no-install       With --build, run ninja without install"
    echo "  --build            Build ASTER (ninja install) too; default only configures + prints build/run cmds"
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

ensure_path_entry() {
    local dir="$1"
    case ":${PATH}:" in
        *:"$dir":*) ;;
        *) export PATH="$dir:$PATH" ;;
    esac
}

USER_LOCAL_BIN="${USER_LOCAL_BIN:-$HOME/.local/bin}"

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
INSTALL_PREREQS=false
LLVM_ONLY=false
HIP_EXPLICIT=""
# ASTER_ENABLE_CPU is an env var, not a CLI flag. Normalise to true/false.
case "${ASTER_ENABLE_CPU:-0}" in
    1|ON|on|true|TRUE|yes|YES) WITH_CPU=true ;;
    *) WITH_CPU=false ;;
esac
ROCM_TARGET_EXPLICIT=""
ROCM_USER_CMD=""
SKIP_ROCM_TEST=true
PRIMARY_GPU_MCPU=""
DETECTED_GPU_MCUS=()
CLANG_CMD="clang"
CLANGXX_CMD="clang++"
CCACHE_CMD=""
LLD_CMD="lld"
VENV_EXPLICIT=""
VENV_PROMPT_EXPLICIT=""
PYTHON_EXPLICIT=""
NO_INSTALL=false
ROCM_PATH_EXPLICIT=""
EXTERNAL_ROCM=false
EXTERNAL_ROCM_DIR=""
EXTERNAL_ROCM_LIB=""
BUILD_ASTER=false
LINKER_CHOICE=""

parse_arguments() {
    for arg in "$@"; do
        case "$arg" in
            --skip-llvm)         SKIP_LLVM=true ;;
            --skip-requirements) SKIP_REQUIREMENTS=true ;;
            --install-prerequisites) INSTALL_PREREQS=true ;;
            --llvm-only)       LLVM_ONLY=true ;;
            --with-hip)        HIP_EXPLICIT=true ;;
            --without-hip)     HIP_EXPLICIT=false ;;
            --rocm-target=*)   ROCM_TARGET_EXPLICIT="${arg#*=}" ;;
            --test-rocm)       SKIP_ROCM_TEST=false ;;
            --clang=*)         CLANG_CMD="${arg#*=}" ;;
            --clang++=*)       CLANGXX_CMD="${arg#*=}" ;;
            --ccache=*)        CCACHE_CMD="${arg#*=}" ;;
            --lld=*)           LLD_CMD="${arg#*=}" ;;
            --python=*)        PYTHON_EXPLICIT="${arg#*=}" ;;
            --venv=*)          VENV_EXPLICIT="${arg#*=}" ;;
            --venv-prompt=*)   VENV_PROMPT_EXPLICIT="${arg#*=}" ;;
            --no-install)      NO_INSTALL=true ;;
            --rocm-path=*)     ROCM_PATH_EXPLICIT="${arg#*=}" ;;
            --build)           BUILD_ASTER=true ;;
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
    # Explicit opt-out always wins.
    if [ "$HIP_EXPLICIT" = "false" ]; then
        WITH_HIP=false
        return
    fi
    # --rocm-path / --with-hip / --rocm-target all mean "set up the ROCm runtime".
    if [ -n "$ROCM_PATH_EXPLICIT" ] || [ "$HIP_EXPLICIT" = "true" ] || [ -n "$ROCM_TARGET_EXPLICIT" ]; then
        WITH_HIP=true
        return
    fi
    # No AMD GPU stack on macOS.
    if [ "$(uname)" = "Darwin" ]; then
        WITH_HIP=false
        return
    fi
    # Linux default: set up ROCm (HIP is dlopen'd at run time).
    WITH_HIP=true
}

phase1_map_mcpu_to_rocm_target() {
    case "$1" in
        gfx1250|gfx1251) echo "" ;;
        gfx940|gfx942)     echo "gfx94X" ;;
        gfx950)            echo "gfx950" ;;
        gfx1200|gfx1201)   echo "gfx120X-all" ;;
        *)                 echo "" ;;
    esac
}

phase1_detect_gpu_mcus() {
    PRIMARY_GPU_MCPU=""
    DETECTED_GPU_MCUS=()
    if ! command -v rocminfo >/dev/null 2>&1; then
        return 0
    fi

    local arch arches=() pref existing found
    while IFS= read -r arch; do
        [ -z "$arch" ] && continue
        arch=${arch%%:*}
        found=false
        for existing in "${arches[@]}"; do
            [ "$existing" = "$arch" ] && { found=true; break; }
        done
        [ "$found" = false ] && arches+=("$arch")
    done <<EOF
$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]{3,4}[a-z0-9]*' || true)
EOF

    if [ ${#arches[@]} -eq 0 ]; then
        return 0
    fi

    DETECTED_GPU_MCUS=("${arches[@]}")
    for pref in gfx1250 gfx1251 gfx950 gfx942 gfx940 gfx1201 gfx1200; do
        for arch in "${DETECTED_GPU_MCUS[@]}"; do
            if [ "$arch" = "$pref" ]; then
                PRIMARY_GPU_MCPU="$arch"
                ok "detected GPU arch: ${DETECTED_GPU_MCUS[*]} (primary: $PRIMARY_GPU_MCPU)"
                return 0
            fi
        done
    done

    PRIMARY_GPU_MCPU="${DETECTED_GPU_MCUS[0]}"
    ok "detected GPU arch: ${DETECTED_GPU_MCUS[*]} (primary: $PRIMARY_GPU_MCPU)"
}

phase1_rocm_target_label() {
    case "$1" in
        gfx1250)     echo "gfx1250/gfx1251 (MI450 / GFX12.5, wave32)" ;;
        gfx94X)      echo "gfx94X (MI300 / CDNA3)" ;;
        gfx950)      echo "gfx950 (MI350 / CDNA4)" ;;
        gfx120X-all) echo "gfx120X-all (RDNA4 consumer)" ;;
        *)           echo "$1" ;;
    esac
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

phase1_prepare_path() {
    ensure_path_entry "$USER_LOCAL_BIN"
    if [ -d "$HOME/.cargo/bin" ]; then
        ensure_path_entry "$HOME/.cargo/bin"
    fi
}

phase1_bootstrap_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    info "Installing uv to $USER_LOCAL_BIN (no sudo)..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$USER_LOCAL_BIN" sh; then
        err "Failed to install uv"
        return 1
    fi
    phase1_prepare_path
    if command -v uv >/dev/null 2>&1; then
        ok "uv installed ($(command -v uv))"
        return 0
    fi
    err "uv install script finished but uv is not on PATH"
    return 1
}

phase1_install_uv_tool() {
    local package="$1"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv is required to install $package locally"
        return 1
    fi
    echo "  Installing $package via uv tool install..."
    if UV_TOOL_BIN_DIR="$USER_LOCAL_BIN" uv tool install "$package"; then
        phase1_prepare_path
        ok "$package installed via uv"
        return 0
    fi
    err "Failed to install $package via uv"
    return 1
}

phase1_install_python_via_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        err "uv is required to install Python locally"
        return 1
    fi
    echo "  Installing Python 3.12 via uv python install..."
    if uv python install 3.12; then
        phase1_prepare_path
        ok "Python 3.12 installed via uv"
        return 0
    fi
    err "Failed to install Python 3.12 via uv"
    return 1
}

phase1_resolve_ccache() {
    if [ -n "$CCACHE_CMD" ]; then
        if [ ! -x "$CCACHE_CMD" ]; then
            err "specified ccache not found or not executable: $CCACHE_CMD"
            CCACHE_CMD=""
            return
        fi
        ok "ccache ($CCACHE_CMD) [--ccache]"
    elif command -v ccache >/dev/null 2>&1; then
        CCACHE_CMD="$(command -v ccache)"
        ok "ccache ($CCACHE_CMD)"
    else
        warn "ccache not found (optional); builds will run without compiler cache"
        warn "  Pass --ccache=PATH if ccache is installed outside PATH"
        return
    fi

    export CCACHE="$CCACHE_CMD"
    export PATH="$(dirname "$CCACHE_CMD"):$PATH"
}

phase1_install_mold() {
    local ver="2.34.1"
    local url="https://github.com/rui314/mold/releases/download/v${ver}/mold-${ver}-x86_64-linux.tar.gz"
    info "Installing mold ${ver} to $HOME/.local (no sudo)..."
    if curl -LsSf "$url" | tar -xz -C "$HOME/.local" --strip-components=1 2>/dev/null; then
        phase1_prepare_path
        if command -v mold >/dev/null 2>&1; then
            ok "mold installed ($(command -v mold))"
            return 0
        fi
    fi
    err "mold install failed (install mold or lld manually for fast linking)"
    return 1
}

# Pick a fast parallel linker on Linux (mold preferred, then lld) and cache it in
# LINKER_CHOICE, used for both the shared-LLVM and ASTER links. Linking with the
# default bfd ld is very slow; mold/lld are dramatically faster. macOS keeps its
# native linker (lld/mold flags can break the Mach-O link).
resolve_parallel_linker() {
    LINKER_CHOICE=""
    [ "$(uname)" != "Linux" ] && return

    # Explicit --lld=PATH wins when it resolves.
    if [ "$LLD_CMD" != "lld" ] && command -v "$LLD_CMD" >/dev/null 2>&1; then
        LINKER_CHOICE="$LLD_CMD"
        ok "parallel linker: $LINKER_CHOICE (--lld)"
        return
    fi

    # mold is the fastest; lld ships with clang (often as ld.lld).
    if command -v mold >/dev/null 2>&1 || command -v ld.mold >/dev/null 2>&1; then
        LINKER_CHOICE="mold"
    elif command -v ld.lld >/dev/null 2>&1 || command -v lld >/dev/null 2>&1; then
        LINKER_CHOICE="lld"
    fi

    if [ -z "$LINKER_CHOICE" ]; then
        warn "No parallel linker (mold/lld) on PATH -- Linux linking will be SLOW (default ld)"
        if [ "$INSTALL_PREREQS" = true ] || ask "Install mold (fast parallel linker, no sudo) now?"; then
            phase1_install_mold && command -v mold >/dev/null 2>&1 && LINKER_CHOICE="mold"
        fi
    fi

    if [ -n "$LINKER_CHOICE" ]; then
        ok "parallel linker: $LINKER_CHOICE"
    else
        warn "continuing without a parallel linker; install mold or lld to speed up linking"
    fi
}

phase1_install_missing_no_sudo() {
    local item installed_any=false needs_uv=false

    for item in "${MISSING[@]}"; do
        [ "$item" = "uv" ] && needs_uv=true
    done
    if [ "$needs_uv" = true ] || ! command -v uv >/dev/null 2>&1; then
        phase1_bootstrap_uv && installed_any=true
    fi
    phase1_prepare_path

    for item in "${MISSING[@]}"; do
        case "$item" in
            uv)
                ;;
            python3*)
                phase1_install_python_via_uv && installed_any=true
                ;;
            cmake)
                phase1_install_uv_tool cmake && installed_any=true
                ;;
            ninja|ninja-build)
                phase1_install_uv_tool ninja && installed_any=true
                ;;
            git)
                warn "git must be installed separately (no sudo auto-install)"
                ;;
            *)
                mapped=$(phase1_map_package "$item")
                if [ -n "$mapped" ] && [ "$mapped" != "$item" ]; then
                    case "$mapped" in
                        cmake) phase1_install_uv_tool cmake && installed_any=true ;;
                        ninja-build|ninja) phase1_install_uv_tool ninja && installed_any=true ;;
                        *) warn "No local installer for $item (package: $mapped)" ;;
                    esac
                else
                    warn "No local installer for $item"
                fi
                ;;
        esac
    done
    [ "$installed_any" = true ]
}

phase1_check_commands() {
    check_required_cmd git
    check_required_cmd cmake
    check_required_cmd ninja
    check_required_cmd "$CLANG_CMD"
    check_required_cmd "$CLANGXX_CMD"
    check_optional_cmd "$LLD_CMD"
    check_required_cmd uv
    phase1_resolve_ccache
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
    echo "  Install missing prerequisites without sudo:"
    echo "    tools/setup.sh --install-prerequisites"
    echo "  Or install manually:"
    [ "$needs_uv" = true ] && \
        echo "    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=\$HOME/.local/bin sh"
    [ "$needs_python" = true ] && \
        echo "    uv python install 3.12"
    for item in "${MISSING[@]}"; do
        case "$item" in
            cmake) echo "    uv tool install cmake" ;;
            ninja|ninja-build) echo "    uv tool install ninja" ;;
        esac
    done
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "  If you have sudo, package-manager installs also work:"
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
    echo ""
}

phase1_prerequisites() {
    info "Phase 1: Checking prerequisites"
    MISSING=()
    phase1_detect_platform
    phase1_prepare_path
    phase1_check_commands
    phase1_resolve_python

    if [ ${#MISSING[@]} -gt 0 ]; then
        if [ "$INSTALL_PREREQS" = true ] || ask "Install missing prerequisites locally (no sudo)?"; then
            phase1_install_missing_no_sudo
            MISSING=()
            phase1_check_commands
            phase1_resolve_python
        fi
    fi

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

    if [ "$LLVM_OK" = true ] && [ -x "$LLVM_INSTALL/bin/llvm-mc" ]; then
        if echo 's_endpgm' | "$LLVM_INSTALL/bin/llvm-mc" \
                -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
                -mattr=+wavefrontsize32 -filetype=obj -o /dev/null 2>/dev/null; then
            ok "shared LLVM llvm-mc supports gfx1250 assembly"
        else
            warn "shared LLVM llvm-mc cannot assemble gfx1250; rebuild shared LLVM"
            LLVM_OK=false
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
    if [ -n "$LINKER_CHOICE" ]; then
        LLVM_LINKER_FLAGS="-DLLVM_USE_LINKER=${LINKER_CHOICE}"
        ok "LLVM link uses ${LINKER_CHOICE} (parallel linker)"
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
    if [ -n "$CCACHE_CMD" ]; then
        export CCACHE="$CCACHE_CMD"
        export PATH="$(dirname "$CCACHE_CMD"):$PATH"
        export LLVM_CCACHE_BUILD=ON
    else
        export LLVM_CCACHE_BUILD=OFF
        warn "ccache disabled for LLVM build"
    fi
    if ! bash "$ASTER_DIR/tools/build-llvm.sh"; then
        err "Shared LLVM build failed"
        echo "  If cmake configure failed partway, retry with:"
        echo "    LLVM_CLEAN_BUILD=ON tools/setup.sh --llvm-only"
        exit 1
    fi
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

    if [ -z "$ROCM_TARGET_EXPLICIT" ] && [ -n "$PRIMARY_GPU_MCPU" ]; then
        ROCM_TARGET_EXPLICIT=$(phase1_map_mcpu_to_rocm_target "$PRIMARY_GPU_MCPU")
        [ -z "$ROCM_TARGET_EXPLICIT" ] && ROCM_TARGET_EXPLICIT=""
        if [ -n "$ROCM_TARGET_EXPLICIT" ] && \
           [ ! -f "$ASTER_DIR/requirements-amd-$ROCM_TARGET_EXPLICIT.txt" ]; then
            ROCM_TARGET_EXPLICIT=""
        elif [ -n "$ROCM_TARGET_EXPLICIT" ]; then
            ok "mapped detected GPU $PRIMARY_GPU_MCPU -> ROCm target $ROCM_TARGET_EXPLICIT"
        fi
    fi

    if [ -n "$ROCM_TARGET_EXPLICIT" ]; then
        ROCM_REQ="$ASTER_DIR/requirements-amd-$ROCM_TARGET_EXPLICIT.txt"
        if [ ! -f "$ROCM_REQ" ]; then
            err "Unknown ROCm target: $ROCM_TARGET_EXPLICIT"
            echo "  Available: gfx94X gfx950 gfx120X-all gfx1250"
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
        echo "  Pass --rocm-target=gfx1250 (or gfx94X, gfx950, gfx120X-all)"
        exit 1
    fi

    # The last menu entry is an escape hatch for a user-supplied install command.
    USER_CMD_CHOICE=$(( ${#ROCM_REQ_FILES[@]} + 1 ))

    echo ""
    echo "  Available ROCm SDK targets:"
    for i in "${!ROCM_REQ_FILES[@]}"; do
        BASENAME=$(basename "${ROCM_REQ_FILES[$i]}" .txt)
        TARGET=${BASENAME#requirements-amd-}
        echo "    $((i+1))) $(phase1_rocm_target_label "$TARGET")"
    done
    echo "    $USER_CMD_CHOICE) User-specified command (i.e. pip install xxx)"
    echo ""
    echo -n "  Which target? [1-$USER_CMD_CHOICE] "
    read -r ROCM_CHOICE

    if ! [[ "$ROCM_CHOICE" =~ ^[0-9]+$ ]] || [ "$ROCM_CHOICE" -lt 1 ] || [ "$ROCM_CHOICE" -gt "$USER_CMD_CHOICE" ]; then
        err "Invalid choice: $ROCM_CHOICE"
        exit 1
    fi

    if [ "$ROCM_CHOICE" -eq "$USER_CMD_CHOICE" ]; then
        echo ""
        echo "  Enter the uv pip install command to run (installs into the ASTER venv):"
        echo "    e.g. uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ --prerelease=allow 'rocm[libraries,devel]'"
        echo ""
        echo -n "  Command: "
        read -r ROCM_USER_CMD
        if [ -z "$ROCM_USER_CMD" ]; then
            err "No command entered"
            exit 1
        fi
        ROCM_TARGET="user"
        ROCM_REQ=""
        return
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

    # User-specified install command: run it verbatim. VIRTUAL_ENV is exported
    # in a subshell so uv pip install targets the ASTER venv even when the
    # command omits an explicit --python.
    if [ -n "$ROCM_USER_CMD" ]; then
        info "Running user-specified ROCm install command"
        echo "    $ROCM_USER_CMD"
        echo ""
        if ( export VIRTUAL_ENV; eval "$ROCM_USER_CMD" ); then
            rm -f "$VIRTUAL_ENV"/.rocm-stamp-* 2>/dev/null
            touch "$VIRTUAL_ENV/.rocm-stamp-user"
            ok "ROCm SDK (user-specified) installed"
        else
            err "User-specified install command failed"
            exit 1
        fi
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

phase3_verify_hip_runtime() {
    # ASTER dlopens libamdhip64.so at run time (no build-time HIP link). "Finding
    # HIP" therefore means the runtime lib is present in the configured ROCm. Fail
    # hard if HIP was requested but it is missing, rather than silently skipping.
    local cand hip_lib=""
    for cand in "$ROCM_PATH/lib/libamdhip64.so" "$ROCM_PATH/lib64/libamdhip64.so"; do
        if [ -e "$cand" ]; then hip_lib="$cand"; break; fi
    done
    if [ -z "$hip_lib" ]; then
        err "HIP requested but libamdhip64.so not found under $ROCM_PATH/{lib,lib64}"
        echo "  GPU execution needs the HIP runtime (ASTER dlopens libamdhip64.so)."
        echo "  For gfx1250 pass --rocm-path=DIR; otherwise check the ROCm SDK install."
        exit 1
    fi
    ok "HIP runtime found: $hip_lib"
}

phase3_use_external_rocm() {
    info "Using preinstalled ROCm at $ROCM_PATH_EXPLICIT (--rocm-path; skipping uv pip install)"
    if [ ! -d "$ROCM_PATH_EXPLICIT" ]; then
        err "--rocm-path directory not found: $ROCM_PATH_EXPLICIT"
        exit 1
    fi
    EXTERNAL_ROCM_DIR="$(cd "$ROCM_PATH_EXPLICIT" && pwd)"

    # Locate libamdhip64.so under lib (or lib64).
    if [ -e "$EXTERNAL_ROCM_DIR/lib/libamdhip64.so" ]; then
        EXTERNAL_ROCM_LIB="$EXTERNAL_ROCM_DIR/lib"
    elif [ -e "$EXTERNAL_ROCM_DIR/lib64/libamdhip64.so" ]; then
        EXTERNAL_ROCM_LIB="$EXTERNAL_ROCM_DIR/lib64"
    else
        err "no libamdhip64.so under $EXTERNAL_ROCM_DIR/{lib,lib64}"
        echo "  --rocm-path must point at a ROCm install containing lib/libamdhip64.so"
        exit 1
    fi
    ok "found HIP runtime: $EXTERNAL_ROCM_LIB/libamdhip64.so"

    export ROCM_PATH="$EXTERNAL_ROCM_DIR"
    export HIP_PATH="$EXTERNAL_ROCM_DIR"
    CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/opt/rocm' | tr '\n' ':' | sed 's/:$//')
    export PATH="$EXTERNAL_ROCM_DIR/bin:$CLEAN_PATH"
    export LD_LIBRARY_PATH="$EXTERNAL_ROCM_LIB:${LD_LIBRARY_PATH:-}"
    EXTERNAL_ROCM=true
    ok "external ROCm configured (ROCM_PATH=$EXTERNAL_ROCM_DIR)"
}

phase3_maybe_setup_rocm() {
    if [ "$WITH_HIP" != true ]; then
        return
    fi

    if [ "$(uname)" = "Darwin" ]; then
        err "--with-hip is only supported on Linux (AMD GPUs require Linux + ROCm)"
        exit 1
    fi

    if [ -n "$ROCM_PATH_EXPLICIT" ]; then
        phase3_use_external_rocm
        phase3_verify_hip_runtime
        echo ""
        return
    fi

    # gfx1250/gfx1251 (MI450) has NO valid pip ROCm: the GFX12 nightlies index is
    # RDNA4 (gfx120X), not MI450. It requires a preinstalled ROCm via --rocm-path.
    if [ "$ROCM_TARGET_EXPLICIT" = "gfx1250" ] || [ "$PRIMARY_GPU_MCPU" = "gfx1250" ] || [ "$PRIMARY_GPU_MCPU" = "gfx1251" ]; then
        err "gfx1250/gfx1251 has no pip ROCm SDK (the GFX12 nightlies are RDNA4 gfx120X, not MI450)"
        echo "  Pass --rocm-path=DIR pointing at a preinstalled ROCm (with lib/libamdhip64.so),"
        echo "  or --without-hip to build cross-compile only (no GPU execution)."
        exit 1
    fi

    phase3_select_rocm_target
    phase3_install_rocm_sdk
    phase3_configure_rocm_env
    phase3_init_and_test_rocm
    phase3_verify_hip_runtime
    echo ""
}

phase3_update_activate_script() {
    ACTIVATE="$VIRTUAL_ENV/bin/activate"
    # Regenerate if the block is missing, doesn't include python_packages, or the
    # ROCm mode changed (external --rocm-path vs pip ROCm SDK).
    if grep -q "python_packages" "$ACTIVATE" 2>/dev/null; then
        if [ "$EXTERNAL_ROCM" = true ] && ! grep -q "ASTER_ROCM_PATH" "$ACTIVATE" 2>/dev/null; then
            : # external ROCm requested but the block is pip-style; regenerate
        else
            ok "activate script already configured"
            return
        fi
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
    printf 'export USER_LOCAL_BIN=%s\n' "$USER_LOCAL_BIN" >> "$ACTIVATE"
    cat >> "$ACTIVATE" << 'ACTIVATE_EOF'
export VENV_PURELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=${VIRTUAL_ENV}/python_packages:${VENV_PURELIB}:${PYTHONPATH}
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
ACTIVATE_EOF
    if [ "$EXTERNAL_ROCM" = true ]; then
        # Preinstalled ROCm on disk (--rocm-path): point at DIR/lib + DIR/bin
        # instead of the pip ROCm SDK under _rocm_sdk_devel.
        printf 'export ASTER_ROCM_PATH=%s\n' "$EXTERNAL_ROCM_DIR" >> "$ACTIVATE"
        printf 'export ASTER_ROCM_LIB=%s\n' "$EXTERNAL_ROCM_LIB" >> "$ACTIVATE"
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'
export ROCM_PATH=${ASTER_ROCM_PATH}
export HIP_PATH=${ASTER_ROCM_PATH}
export PATH=${USER_LOCAL_BIN}:${LLVM_INSTALL}/bin:${VIRTUAL_ENV}/bin:${ASTER_ROCM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${ASTER_ROCM_LIB}:${LD_LIBRARY_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF
    else
        cat >> "$ACTIVATE" << 'ACTIVATE_EOF'
export PATH=${USER_LOCAL_BIN}:${LLVM_INSTALL}/bin:${VIRTUAL_ENV}/bin:${VENV_PURELIB}/_rocm_sdk_devel/bin:${PATH}
export LD_LIBRARY_PATH=${VENV_PURELIB}/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH}
# --- end ASTER setup ---
ACTIVATE_EOF
    fi

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
    # ASTER has NO build-time HIP/ROCm dependency: the HIP runtime is dlopen'd
    # (libamdhip64.so) at run time and verified during ROCm setup (phase3). The
    # build needs nothing from ROCm at cmake time, so we do NOT find_package(hip)
    # or pass HIP_PLATFORM (nothing consumes it; it only litters the cmake cache
    # as HIP_PLATFORM:UNINITIALIZED).
    CMAKE_EXTRA_FLAGS=""
    CMAKE_PREFIX_CHAIN="$LLVM_INSTALL"
    if [ "$WITH_HIP" = true ]; then
        ok "ROCm runtime configured (HIP dlopen'd at run time; no build-time HIP dep)"
    else
        ok "No ROCm runtime (cross-compile mode, no GPU execution)"
    fi
}

phase4_needs_reconfigure() {
    NEED_RECONFIGURE=false
    if [ ! -f "$ASTER_BUILD_DIR/CMakeCache.txt" ] || [ ! -f "$ASTER_BUILD_DIR/build.ninja" ]; then
        NEED_RECONFIGURE=true
    fi
}

phase4_select_linker() {
    ASTER_LINKER_FLAGS=""
    if [ -n "$LINKER_CHOICE" ]; then
        ASTER_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=${LINKER_CHOICE} -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=${LINKER_CHOICE} -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=${LINKER_CHOICE}"
        ok "ASTER link uses ${LINKER_CHOICE} (parallel linker)"
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
    echo "  build:   $ASTER_BUILD_DIR (cmake configured)"
    [ -n "$LINKER_CHOICE" ] && echo "  linker:  $LINKER_CHOICE (parallel)"
    echo ""
    if [ "$BUILD_ASTER" = true ]; then
        ok "ASTER built (--build)."
        echo ""
        echo "  Activate, then run:"
        echo "    source $VIRTUAL_ENV/bin/activate"
        echo "    lit $ASTER_BUILD_DIR/test -v"
        echo "    cd $ASTER_DIR && pytest -n 16 ./test ./mlir_kernels ./contrib ./python"
        echo "    Rebuild: ninja -C $ASTER_BUILD_DIR install"
    else
        echo "  Prerequisites configured; ASTER was NOT built (configure-only default)."
        echo "  Activate, build, and run:"
        echo ""
        echo "    1) Activate the venv:"
        echo "         source $VIRTUAL_ENV/bin/activate"
        echo "    2) Build ASTER:"
        echo "         ninja -C $ASTER_BUILD_DIR install"
        echo "    3) Run tests:"
        echo "         lit $ASTER_BUILD_DIR/test -v"
        echo "         cd $ASTER_DIR && pytest -n 16 ./test ./mlir_kernels ./contrib ./python"
        echo ""
        echo "  (Re-run tools/setup.sh --build to configure AND build in one step.)"
    fi
}

main() {
    parse_arguments "$@"
    resolve_virtual_env
    phase1_detect_gpu_mcus
    resolve_with_hip

    phase1_prerequisites
    resolve_parallel_linker
    phase2_shared_llvm

    if [ "$LLVM_ONLY" = true ]; then
        info "Done (--llvm-only). Shared LLVM is ready at $LLVM_INSTALL"
        exit 0
    fi

    phase3_python_venv
    phase4_cmake_configure
    if [ "$BUILD_ASTER" = true ]; then
        phase5_build
    fi
    print_summary
}

main "$@"
