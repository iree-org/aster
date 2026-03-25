# Developer Setup: Worktrees and Shared LLVM

## Shared LLVM Build

Build LLVM once in a central location, share across all worktrees. Avoids rebuilding LLVM (90%+ of build time) per worktree.

| Path | Purpose |
|------|---------|
| `${HOME}/shared-llvm` | Shared LLVM install prefix |
| `${HOME}/llvm-build` | LLVM build directory (can delete after install) |

### One-time setup cost: Building shared LLVM

Clone the LLVM project at the pinned commit and build it:

```bash
LLVM_COMMIT=$(cat llvm/LLVM_COMMIT)
git clone https://github.com/nicolasvasilache/llvm-project.git ${HOME}/llvm-project
git -C ${HOME}/llvm-project checkout ${LLVM_COMMIT}

export LLVM_SRC=${HOME}/llvm-project/llvm
export LLVM_INSTALL=${HOME}/shared-llvm
export LLVM_BUILD=${HOME}/llvm-build

# MLIR recommended setup for python bindings
export LLVM_VENV=${LLVM_BUILD}/.venv
uv venv ${LLVM_VENV} --seed -p 3.12
source ${LLVM_VENV}/bin/activate
uv pip install -r ${LLVM_SRC}/../mlir/python/requirements.txt

mkdir -p "$LLVM_BUILD" && cd "$LLVM_BUILD"

cmake "$LLVM_SRC" -GNinja \
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
  -DPython_EXECUTABLE="${LLVM_VENV}/bin/python" \
  -DPython3_EXECUTABLE="${LLVM_VENV}/bin/python" \
  -DLLVM_CCACHE_BUILD=ON

# On Linux with lld or mold available, add for faster link times:
#   -DLLVM_USE_LINKER=lld
# On macOS the system linker is already fast so this is not needed.

# On Linux with ROCm SDK installed (optional, for HIP-aware LLVM):
#   -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
#   -DHIP_PLATFORM=amd

ninja install

# Install test tools
ninja install FileCheck count not llvm-objdump

# Note: on some systems the LLVM CMake does not install those tools properly so
# one may need to manually copy them:
cp ${LLVM_BUILD}/bin/FileCheck ${LLVM_INSTALL}/bin/FileCheck
cp ${LLVM_BUILD}/bin/count ${LLVM_INSTALL}/bin/count
cp ${LLVM_BUILD}/bin/not ${LLVM_INSTALL}/bin/not
cp ${LLVM_BUILD}/bin/llvm-objdump ${LLVM_INSTALL}/bin/llvm-objdump
```

Rebuild when `llvm/LLVM_COMMIT` is updated or you need different build options:

```bash
export LLVM_BUILD=${HOME}/llvm-build
export LLVM_VENV=${LLVM_BUILD}/.venv
export LLVM_INSTALL=${HOME}/shared-llvm
LLVM_COMMIT=$(cat llvm/LLVM_COMMIT)
git -C ${HOME}/llvm-project fetch origin
git -C ${HOME}/llvm-project checkout ${LLVM_COMMIT}
cd ${LLVM_BUILD} && ninja install
```

If you need to modify LLVM and build test it:

```bash
export LLVM_BUILD=${HOME}/llvm-build
export LLVM_VENV=${LLVM_BUILD}/.venv
export LLVM_INSTALL=${HOME}/shared-llvm
deactivate
source ${LLVM_VENV}/bin/activate

# build mlir-opt and runn all tests
cd ${HOME}/llvm-build && ninja mlir-opt && ninja check-mlir

# run single test
${HOME}/llvm-build/bin/llvm-lit ~/llvm-project/mlir/test/Dialect/Affine/decompose-affine-ops-cse-friendly.mlir -v
```

### LLVM target support

ASTER has early .hsaco generation support for the following targets, which all
require an appropriate LLVM AMDGPU backend for translating asm to binary:

| Target   | ISA   | Product Family          |
|----------|-------|-------------------------|
| gfx940   | CDNA3 | MI300A                  |
| gfx942   | CDNA3 | MI300X                  |
| gfx950   | CDNA4 | MI350X                  |
| gfx1201  | RDNA4 | Radeon RX 9070          |

HSACO assembly (the `assemble_to_hsaco` step) requires the LLVM version to
recognize the target chip. If your LLVM build does not support a given target
(e.g. gfx950 requires a recent LLVM with CDNA4 support), the HSACO step will
be skipped. ASTER's own IR translation will work regardless of LLVM version.

## Manual Setup

`tools/setup.sh` handles all of the below automatically. The manual steps are
here as a reference for customized setups.

### venv

```bash
uv venv .aster --seed --python 3.12 --prompt aster
source .aster/bin/activate
uv pip install -r requirements.txt
```

### Set useful variables in the venv activation script

```bash
LLVM_INSTALL="${HOME}/shared-llvm"
ASTER_DIR="$(pwd)"
cat >> .aster/bin/activate << EOF

# --- ASTER setup (added by tools/setup.sh) ---
export LLVM_INSTALL=${LLVM_INSTALL}
export ASTER_SRC_DIR=${ASTER_DIR}
export VENV_PURELIB=\$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PATH=\${LLVM_INSTALL}/bin:\${VIRTUAL_ENV}/bin:\${VENV_PURELIB}/_rocm_sdk_devel/bin:\${PATH}
export PYTHONPATH=\${VIRTUAL_ENV}/python_packages:\${VENV_PURELIB}:\${PYTHONPATH}
export LD_LIBRARY_PATH=\${VENV_PURELIB}/_rocm_sdk_devel/lib:\${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=\${LLVM_INSTALL}:\${CMAKE_PREFIX_PATH}
# --- end ASTER setup ---
EOF

deactivate
source .aster/bin/activate
```

### Building ASTER (macOS and Linux without GPU)

This builds ASTER for cross-compilation only (no HIP runtime, no on-device
execution). If you have a stale `build/CMakeCache.txt` from a previous
configure (e.g. with a different venv), delete it first: `rm build/CMakeCache.txt`.

```bash
(
  mkdir -p build && cd build

  CMAKE_PREFIX_PATH="${LLVM_INSTALL}" \
  cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -DLLVM_EXTERNAL_LIT="${VIRTUAL_ENV}/bin/lit" \
    -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
    -DPython3_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=aster

  # On Linux, add for faster link times:
  #   -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld
  #   -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld
  #   -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld
  # On macOS use the default system linker (no lld needed).

  ninja install
)
```

### Linux with AMD GPU support

For HIP runtime support and execution tests on actual hardware, install
[theRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) which provides
ROCm as a Python package:

```bash
# Choose based on your GPU architecture:

# For RDNA4 (gfx120x):
uv pip install -r requirements-amd-gfx120X-all.txt

# For CDNA3 (MI300, gfx94x):
uv pip install -r requirements-amd-gfx94X.txt

# For CDNA4 (MI350, gfx950):
uv pip install -r requirements-amd-gfx950.txt

# Initialize and test ROCm
rocm-sdk init
rocm-sdk test
```

Then build with HIP support (add these flags to the cmake command above):

```bash
    -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
    -DHIP_PLATFORM=amd
```

## Worktree Setup

Each worktree needs its own build directory and venv, but shares LLVM.
The simplest way is `tools/setup.sh --skip-llvm` run from the worktree directory.

For manual setup:

### venv

```bash
cd /path/to/worktree

deactivate ; unset PYTHONPATH  # clear any active venv state
uv venv .aster --seed --python 3.12 --prompt aster
source .aster/bin/activate
uv pip install -r requirements.txt
```

### Set useful variables in the venv activation script

Same as the main repo setup above — the venv is always named `.aster`.

### Building with shared LLVM

```bash
(
  mkdir -p build && cd build

  CMAKE_PREFIX_PATH="${LLVM_INSTALL}" \
  cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -DLLVM_EXTERNAL_LIT="${VIRTUAL_ENV}/bin/lit" \
    -DPython_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
    -DPython3_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=aster

  # On Linux, add for faster link times:
  #   -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld
  #   -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld
  #   -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld
  # On macOS use the default system linker (no lld needed).

  # On Linux with ROCm SDK installed, add:
  #   -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip"
  #   -DHIP_PLATFORM=amd

  ninja install
)
```

First build after cmake configure is fast since LLVM is pre-built.

### Testing

```bash
source .aster/bin/activate

# All tests (lit + pytest)
ninja -C build install && lit build/test -v && pytest -n 16

# Lit tests only (IR roundtrip + ASM checks, includes integration/)
lit build/test -v

# Pytest only (execution on GPU)
pytest -n 16

# Single lit test
lit build/test/integration/conversion-pack-e2e.mlir -s -v

# Single pytest file
pytest test/integration/test_mfma_e2e.py -s -v
```

Test paths (`test/`, `mlir_kernels/`, `contrib/`, `python/`) are configured in
`pyproject.toml` so bare `pytest` discovers everything.

Integration tests in `test/integration/` have both lit RUN directives (ASM verification)
and pytest files (GPU execution). Lit tests run cross-platform; pytest requires a GPU.

## Notes

- Linker: On Linux, use `lld` for LLVM builds and `lld`/`mold` for ASTER builds
  (link times drop from minutes to seconds). On macOS use the default system linker.
- ccache: Never clean it (incremental builds)
- Each worktree has its own `build/` and `.aster/` directories
- All worktrees use the same `${HOME}/shared-llvm`
- Make sure shared LLVM exists and is up to date: `ls ${HOME}/shared-llvm/lib/cmake/llvm`

## Misc: Quick git worktree primer

Git worktrees allow multiple branches checked out simultaneously in separate directories, sharing the same `.git` repository. Useful for working on multiple features/fixes in parallel without stashing or switching branches, and for testing changes across branches without rebuilding everything.

```bash
# List existing worktrees
git worktree list

# Create new worktree from existing branch
git worktree add /path/to/worktree branch-name

# Create new worktree with new branch from on top of another branch (default: main)
git worktree add -b new-branch /path/to/worktree [base-branch-to-start-from]

# Remove worktree
git worktree remove /path/to/worktree

# Prune stale worktree references
git worktree prune
```
