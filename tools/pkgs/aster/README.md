# aster Python wheel packaging

This directory contains the wheel build infrastructure for the `aster` Python package.

| File | Purpose |
|---|---|
| `build.sh` | Orchestrates a `cibuildwheel` run; stages the LLVM tarball |
| `before-all.sh` | `cibuildwheel` hook; extracts LLVM and installs build tools inside the container |

Build system and package metadata live in the root [`pyproject.toml`](../../../pyproject.toml).

---

## Editable install (local development)

Requires `CMAKE_PREFIX_PATH` to point to the LLVM cmake directory so that `find_package(MLIR)` succeeds.

```bash
# Activate the project sandbox first
source <worktree>/sandbox/bin/activate_sandbox

# Install (builds C++ extensions via scikit-build-core + CMake)
CMAKE_PREFIX_PATH=~/shared-llvm/lib/cmake pip install -e .

# With dev extras (lit, pytest, black, pyright, …)
CMAKE_PREFIX_PATH=~/shared-llvm/lib/cmake pip install -e .[dev]
```

To pass extra CMake flags (e.g. override build type or set a custom install prefix):

```bash
CMAKE_PREFIX_PATH=~/shared-llvm/lib/cmake \
  pip install -e . \
  --config-settings=cmake.build-type=Debug \
  --config-settings="cmake.args=-DASTER_ENABLE_UNIT_TESTS=ON"
```

After installation `import aster` works from any directory without modifying `PYTHONPATH`.

---

## Release wheels (cibuildwheel)

`build.sh` stages the LLVM installation as a tarball in the project root and runs `cibuildwheel` from there. Docker is required on Linux.

### Building the LLVM tarball

A pre-built LLVM tarball is required. See [`tools/pkgs/llvm/README.md`](../llvm/README.md) for full details. Quick start:

```bash
# Linux (requires Docker); produces llvm-linux-x86_64-<commit>.tar.gz
bash tools/pkgs/llvm/build.sh --output=/tmp/llvm-out

# macOS (native); produces llvm-macos-arm64-<commit>.tar.gz
bash tools/pkgs/llvm/build.sh --output=/tmp/llvm-out
```

### From a pre-built LLVM tarball

```bash
bash tools/pkgs/aster/build.sh --llvm-tarball=~/llvm-out/llvm-linux-x86_64-<commit>.tar.gz
```

### From an unpacked LLVM directory

`build.sh` re-archives it into the expected `llvm/` layout automatically.

```bash
bash tools/pkgs/aster/build.sh --llvm-dir=~/shared-llvm
```

Wheels land in `./wheelhouse/` by default.

### Common options

| Option | Default | Description |
|---|---|---|
| `--llvm-tarball=PATH` | — | Pre-built LLVM `.tar.gz` |
| `--llvm-dir=PATH` | — | Unpacked LLVM installation directory |
| `--output=DIR` | `./wheelhouse` | Destination for produced wheels |
| `--arch=ARCH` | host arch | `x86_64`, `aarch64`, or `arm64` |
| `--jobs=N` | — | Parallelism forwarded to the build |

### Cross-compilation (aarch64 on x86_64)

Requires Docker with QEMU binfmt support (`docker run --platform linux/arm64` must work).

```bash
bash tools/pkgs/aster/build.sh \
  --llvm-tarball=~/llvm-out/llvm-linux-aarch64-<commit>.tar.gz \
  --arch=aarch64
```

---

## Tarball layout

`build.sh` and `before-all.sh` expect the LLVM tarball to have a top-level `llvm/` directory produced by `tools/pkgs/llvm/build.sh`:

```
llvm/
  bin/
  include/
  lib/
    cmake/
      llvm/
      mlir/
      lld/
```

`before-all.sh` locates `lib/cmake` inside the extracted tree and symlinks it to `/opt/llvm` (Linux) or `~/.llvm-pkg` (macOS) so CMake's `find_package()` can resolve it via `CMAKE_PREFIX_PATH`.
