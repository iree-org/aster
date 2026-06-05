# Developer Setup: Worktrees and Shared LLVM

## Quick start

`tools/setup.sh` does everything -- it builds the shared LLVM, creates the
`.aster` venv, configures cmake, and builds ASTER. It is idempotent (safe to
re-run) and works on macOS and Linux.

```bash
# macOS, or Linux without a GPU (cross-compile only):
bash tools/setup.sh

# Linux with an AMD GPU (installs the ROCm SDK and builds with HIP):
bash tools/setup.sh --with-hip --rocm-target=gfx94X   # or gfx950, gfx120X

# Inside an existing worktree (reuse the shared LLVM, do not rebuild it):
bash tools/setup.sh --skip-llvm

# Only (re)build the shared LLVM:
bash tools/setup.sh --llvm-only
```

Run `bash tools/setup.sh --help` for all options. The rest of this document is
reference material for manual or customized setups.

## Shared LLVM Build

Build LLVM once in a central location, shared across all worktrees. This avoids
rebuilding LLVM (90%+ of build time) per worktree.

| Path | Purpose |
|------|---------|
| `${HOME}/shared-llvm` | Shared LLVM install prefix |
| `${HOME}/llvm-build` | LLVM build directory (can delete after install) |

`tools/setup.sh --llvm-only` clones the pinned LLVM (`llvm/LLVM_COMMIT`), sets up
its python venv, and builds + installs it to `${HOME}/shared-llvm` with the right
MLIR/AMDGPU options (and the `FileCheck`/`count`/`not`/`llvm-objdump` test tools).
Re-run it after `llvm/LLVM_COMMIT` changes. The script is the source of truth for
the exact cmake flags.

### Modifying LLVM itself

To modify LLVM and build-test it incrementally:

```bash
export LLVM_BUILD=${HOME}/llvm-build
export LLVM_VENV=${LLVM_BUILD}/.venv
export LLVM_INSTALL=${HOME}/shared-llvm
deactivate
source ${LLVM_VENV}/bin/activate

# build mlir-opt and run all tests
cd ${HOME}/llvm-build && ninja mlir-opt && ninja check-mlir

# run a single test
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

`tools/setup.sh` handles all of the below automatically. These steps are kept as
a reference for customized setups.

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
execution). If you have a stale `build/CMakeCache.txt` from a previous configure
(e.g. with a different venv), delete it first: `rm build/CMakeCache.txt`.

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

  ninja install
)
```

### Linux with AMD GPU support

For HIP runtime support and execution tests on actual hardware, install
[theRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) which provides
ROCm as a Python package (or just use `tools/setup.sh --with-hip`):

```bash
# Choose based on your GPU architecture:
uv pip install -r requirements-amd-gfx120X-all.txt   # RDNA4 (gfx120x)
uv pip install -r requirements-amd-gfx94X.txt        # CDNA3 (MI300, gfx94x)
uv pip install -r requirements-amd-gfx950.txt        # CDNA4 (MI350, gfx950)

# Initialize and test ROCm
rocm-sdk init
rocm-sdk test
```

Then add these flags to the cmake command above:

```bash
    -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
    -DHIP_PLATFORM=amd
```

## Worktree Setup

Each worktree needs its own build directory and venv, but shares the central
LLVM. The simplest way is `tools/setup.sh --skip-llvm` from the worktree
directory. The manual steps mirror the main-repo Manual Setup above -- the venv
is always named `.aster` and the cmake command is identical; only the build
directory and venv are per-worktree.

## Optional: Building with StinkyTofu

ASTER can build AMD StinkyTofu's `stinkytofu-opt` (a gfx1250 assembly-kernel
optimizer) as a contrib tool, to connect the two systems. This work lives on the
`stinkytofu` branch:

```bash
git checkout stinkytofu
```

Then follow `contrib/stinkytofu/README.md` to bootstrap the vendored submodule
and build with `-DASTER_ENABLE_STINKYTOFU=ON`. The contrib is OFF by default and
does not affect the standard ASTER build.

## Testing

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
