# Developer Setup: Worktrees and Shared LLVM

## Shared LLVM Build

Build LLVM once in a central location, share across all worktrees. Avoids rebuilding LLVM (90%+ of build time) per worktree.

| Path | Purpose |
|------|---------|
| `${HOME}$/shared-llvm` | Shared LLVM install prefix |
| `${HOME}$/llvm-build` | LLVM build directory (can delete after install) |

### Building shared LLVM

```bash
LLVM_SRC=${HOME}$/aster-cursor/llvm/llvm-project/llvm
LLVM_INSTALL=${HOME}$/shared-llvm
LLVM_BUILD=${HOME}$/llvm-build

mkdir -p "$LLVM_BUILD" && cd "$LLVM_BUILD"

cmake "$LLVM_SRC" -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DLLVM_CCACHE_BUILD=ON

ninja install

# Install test tools
ninja FileCheck count not llvm-objdump
cp bin/FileCheck bin/count bin/not bin/llvm-objdump "$LLVM_INSTALL/bin/"
```

Rebuild when LLVM submodule is updated (`git submodule status`) or different build options needed.
All worktrees must use the same LLVM submodule commit.

## Worktree Setup

Each worktree needs its own build directory and venv, but shares LLVM.

### venv

```bash
cd /path/to/worktree
python3 -m venv --prompt aster .aster
source .aster/bin/activate
pip install -r requirements.txt
```

### Set useful variables in a Python virtual environment

```bash
cat >> .aster/bin/activate << 'EOF'

export PATH=${PWD}/.aster/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/bin/:${PATH}

export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/lib
EOF

deactivate && source .aster/bin/activate
```

### Building with shared LLVM

```bash
LLVM_INSTALL=${HOME}$/shared-llvm

mkdir -p build && cd build

cmake ../ -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="../.aster" \
  -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT=${VIRTUAL_ENV}/bin/lit \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
  -DHIP_PLATFORM=amd

ninja install
```

First build after cmake configure is fast since LLVM is pre-built.

### Testing

```bash
# Activate venv
source .aster/bin/activate

# Lit tests
(cd build && ninja install) && lit build/test -v

# Pytest
(cd build && ninja install) && pytest -n 16 ./integration_test/

# All tests
(cd build && ninja install) && lit build/test -v && pytest -n 16 ./integration_test/
```

## Notes

- ccache: Never clean it (incremental builds)
- Each worktree has own `build/` and `.aster/` directories
- All worktrees use same `${HOME}$/shared-llvm`
- Verify shared LLVM exists: `ls ${HOME}$/shared-llvm/lib/cmake/llvm`
