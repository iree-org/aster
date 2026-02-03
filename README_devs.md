# Developer Setup: Worktrees and Shared LLVM

## Quick git worktree primer

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



## Shared LLVM Build

Build LLVM once in a central location, share across all worktrees. Avoids rebuilding LLVM (90%+ of build time) per worktree.

| Path | Purpose |
|------|---------|
| `${HOME}/shared-llvm` | Shared LLVM install prefix |
| `${HOME}/llvm-build` | LLVM build directory (can delete after install) |

### On-time setup cost: Building shared LLVM

```bash
export LLVM_SRC=${HOME}/aster/llvm/llvm-project/llvm
export LLVM_INSTALL=${HOME}/shared-llvm
export LLVM_BUILD=${HOME}/llvm-build

mkdir -p "$LLVM_BUILD" && cd "$LLVM_BUILD"

# MLIR recommended setup for python bindings
export LLVM_VENV=${LLVM_BUILD}/.venv
uv venv ${LLVM_VENV} --seed -p 3.12
source ${LLVM_VENV}/bin/activate
uv pip install -r ${LLVM_SRC}/mlir/python/requirements.txt

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
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
  -DHIP_PLATFORM=amd \
  -DLLVM_CCACHE_BUILD=ON

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

Rebuild when LLVM submodule is updated (`git submodule status`) or different build options needed.
All worktrees must use the same LLVM submodule commit.

## Worktree Setup

Each worktree needs its own build directory and venv, but shares LLVM.
We use `uv` to pip install in the new venv, the latency of pure `pip` being too high.

### venv

```bash
cd /path/to/worktree

export WORKTREE_NAME=$(basename $(pwd))
deactivate ; unset PYTHONPATH # in case IDE / bash auto-sets these
python3 -m venv --prompt aster-wt-${WORKTREE_NAME} .aster-wt-${WORKTREE_NAME}
source .aster-wt-${WORKTREE_NAME}/bin/activate
uv pip install -r requirements.txt
```

### Set useful variables in a Python virtual environment

```bash
export WORKTREE_NAME=$(basename $(pwd))
cat >> .aster-wt-${WORKTREE_NAME}/bin/activate << 'EOF'

export WORKTREE_NAME=$(basename $(pwd))

export PATH=${PWD}/.aster-wt-${WORKTREE_NAME}/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/bin/:${PATH}

export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster-wt-${WORKTREE_NAME}/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/lib

export LLVM_SRC=${HOME}/aster/llvm/llvm-project/llvm
export LLVM_INSTALL=${HOME}/shared-llvm
export LLVM_BUILD=${HOME}/llvm-build
EOF

deactivate ;  unset PYTHONPATH; source .aster-wt-${WORKTREE_NAME}/bin/activate
```

### Building with shared LLVM

```bash
(
  mkdir -p build && cd build;

  cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="../.aster-wt-${WORKTREE_NAME}" \
    -DASTER_EXTERNAL_LLVM=ON \
    -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" \
    -DLLD_DIR="$LLVM_INSTALL/lib/cmake/lld" \
    -DLLVM_EXTERNAL_LIT=${VIRTUAL_ENV}/bin/lit \
    -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
    -DHIP_PLATFORM=amd;

  ninja install;
)
```

First build after cmake configure is fast since LLVM is pre-built.

### Testing

```bash
# Activate venv
export WORKTREE_NAME=$(basename $(pwd))
deactivate ;  unset PYTHONPATH; source .aster-wt-${WORKTREE_NAME}/bin/activate

# All tests
(cd build && ninja install) && lit build/test -v && pytest -n 16 ./test ./integration_test ./mlir_kernels

# Lit tests only
(cd build && ninja install) && lit build/test -v

# Pytest only
(cd build && ninja install) && pytest -n 16 ./test ./integration_test ./mlir_kernels
```

## Notes

- ccache: Never clean it (incremental builds)
- Each worktree has own `build/` and `.aster-wt-${WORKTREE_NAME}/` directories
- All worktrees use same `${HOME}/shared-llvm`
- Make sure shared LLVM exists and is up to date: `ls ${HOME}/shared-llvm/lib/cmake/llvm`
