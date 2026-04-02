# LLVM Packaging

Scripts to build, patch, and package LLVM into a portable `tar.gz` that can
be distributed and used without recompiling LLVM from source.

The pinned LLVM commit is read from `llvm/LLVM_COMMIT` at the root of this
repository.  The build configuration (targets, projects, distribution
components) is driven by `tools/build-llvm.sh`.

## Produced artifact

```
llvm-<os>-<arch>-<short-commit>.tar.gz
└── llvm/
    ├── bin/          # mlir-opt, mlir-translate, mlir-tblgen, FileCheck, lld, …
    ├── include/      # LLVM / MLIR / LLD headers
    ├── lib/          # Static libraries (.a)
    └── lib/cmake/    # CMake config files (relocatable, use CMAKE_PREFIX_PATH)
```

The archive is relocatable: extract it anywhere and point `CMAKE_PREFIX_PATH`
at the `llvm/` directory.

## Usage

```bash
bash tools/pkgs/llvm/build.sh [OPTIONS]

Options:
  --output=DIR       Directory to write the tar.gz.          [default: .]
  --llvm-src=DIR     Reuse an existing llvm-project clone.   [default: clone]
  --jobs=N           Parallel build jobs.      [default: nproc / hw.logicalcpu]
  --arch=ARCH        Linux only: x86_64 or aarch64.          [default: host]
  --help             Print this help.
```

**Linux** (requires Docker):

```bash
bash tools/pkgs/llvm/build.sh --output=/tmp/llvm-out
# Produces: /tmp/llvm-out/llvm-linux-x86_64-e98686f.tar.gz
```

**macOS** (native, requires cmake, ninja, clang, git):

```bash
bash tools/pkgs/llvm/build.sh --output=/tmp/llvm-out
# Produces: /tmp/llvm-out/llvm-macos-arm64-e98686f.tar.gz
```

Reuse an existing LLVM checkout to skip the ~500 MB clone:

```bash
bash tools/pkgs/llvm/build.sh --llvm-src=~/llvm-project --output=/tmp/llvm-out
```

## Consuming the package

```cmake
# In your CMakeLists.txt, before find_package calls:
list(PREPEND CMAKE_PREFIX_PATH "/path/to/llvm")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(LLD  REQUIRED CONFIG)
```

Or on the cmake command line:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/llvm ...
```

## Platform portability

### Linux

The build runs inside a
[manylinux_2_28](https://github.com/pypa/manylinux) Docker container
(AlmaLinux 8 based).  This guarantees that the produced binaries depend only
on **glibc ≥ 2.28**, which covers virtually all Linux distributions released
since 2018 (RHEL 8, Ubuntu 20.04+, Debian 10+, etc.).

The C++ runtime (`libstdc++`, `libgcc_s`) is linked **statically** via
`CMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"`, so there is no
runtime dependency on the host C++ standard library.  `patchelf` is used as a
safety net to set a portable `$ORIGIN`-relative RPATH on any shared object
that ends up in the install tree.

Both `x86_64` and `aarch64` are supported:

```bash
bash tools/pkgs/llvm/build.sh --arch=x86_64  --output=/tmp/llvm-out
bash tools/pkgs/llvm/build.sh --arch=aarch64 --output=/tmp/llvm-out
```

### macOS

The build runs natively with `MACOSX_DEPLOYMENT_TARGET=12.0`, targeting
macOS Monterey and later.  `libc++` ships with macOS itself so no bundling is
needed for the C++ runtime.  [delocate](https://github.com/matthew-brett/delocate)
is invoked after the install step to bundle any non-system dylibs into the
package (installed automatically via `pip install delocate` if absent).

Both Apple Silicon (`arm64`) and Intel (`x86_64`) are supported.

## Directory layout

```
tools/pkgs/llvm/
├── build.sh                  # Entry point (dispatches to linux/ or macos/)
├── common/
│   └── cpack-config.cmake.in # CPack TGZ configuration template
├── linux/
│   ├── Dockerfile            # manylinux_2_28 image with clang + patchelf
│   └── entrypoint.sh         # Executed inside the Docker container
└── macos/
    └── build.sh              # Native macOS build script
```

## Build configuration

All cmake flags are inherited from `tools/build-llvm.sh`.  The packaging
scripts set the following env vars before invoking it:

| Variable | Value | Reason |
|---|---|---|
| `LLVM_ENABLE_ASSERTIONS` | `ON` | Required by aster |
| `CMAKE_BUILD_TYPE` | `Release` | Optimised, smaller binaries |
| `LLVM_CCACHE_BUILD` | `OFF` | Not available / not useful in CI / Docker |
| `CMAKE_EXE_LINKER_FLAGS` | `-static-libstdc++ -static-libgcc` | Linux only — removes libstdc++ runtime dep |
| `MACOSX_DEPLOYMENT_TARGET` | `12.0` | macOS only — minimum supported version |
