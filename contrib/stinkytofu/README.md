# contrib/stinkytofu

Build an ASTER <-> StinkyTofu connect and the `stinkytofu-opt` tool with a pinned
submodule from ASTER's CMake.

StinkyTofu source is its own source of truth and is never modified by ASTER.
ASTER provides build shims (comgr stub, no-op ClangTidy, `linux/limits.h`) to
allow cross-compilation and asm generation on non-linux targets.

## Layout

```
contrib/stinkytofu/
  CMakeLists.txt              # wires the vendored tree into ASTER's build
  cmake/ClangTidy.cmake       # no-op shadow of the monorepo's ClangTidy module
  compat/linux/limits.h       # macOS shim (StinkyTofu includes <linux/limits.h>)
  fake_comgr/                 # stub amd_comgr (header + lib + CONFIG package)
  test/                       # lit round-trip test
  third_party/stinkytofu/     # submodule: rocm-libraries (sparse: shared/stinkytofu)
```

## Bootstrap the submodule

StinkyTofu has no standalone repository -- it is the `shared/stinkytofu` subtree
of the `ROCm/rocm-libraries` monorepo (a multi-GB, SSO-gated repo). To avoid
pulling that for everyone, the submodule is committed with a local default `url`
and `update = none`, so a normal `git clone --recursive` / `git submodule update
--init` SKIPS it.

Instead, fetch rocm-libraries yourself and point the submodule at your local
clone (per-user `.git/config`):

```bash
# 1. Obtain rocm-libraries however you like (it is NOT auto-fetched).
git clone https://github.com/ROCm/rocm-libraries.git /path/to/rocm-libraries

# 2. Point the submodule at your local clone (per-user, not committed):
git config submodule.contrib/stinkytofu/third_party/stinkytofu.url /path/to/rocm-libraries

# 3. Populate it (force checkout to override `update = none` for this command):
git -c submodule."contrib/stinkytofu/third_party/stinkytofu".update=checkout \
    submodule update --init --force contrib/stinkytofu/third_party/stinkytofu

# 4. Limit the working tree to StinkyTofu and pin the reviewed commit
#    (the pin is recorded in contrib/stinkytofu/COMMIT):
git -C contrib/stinkytofu/third_party/stinkytofu sparse-checkout init --cone
git -C contrib/stinkytofu/third_party/stinkytofu sparse-checkout set shared/stinkytofu
git -C contrib/stinkytofu/third_party/stinkytofu checkout "$(cat contrib/stinkytofu/COMMIT)"
```


## Build (macOS or Linux)

Stinkytofu is OFF by default and can be enabled it through `tools/setup.sh`
(see the main `README_devs.md` for the full build flow):

```bash
# From the worktree root, reusing the shared LLVM:
bash tools/setup.sh --skip-llvm --with-stinkytofu
```

`--with-stinkytofu` adds `-DASTER_ENABLE_STINKYTOFU=ON` to the cmake configure and
builds the `stinkytofu-opt` tool.

Since comgr is only used by StinkyTofu's assembler-capability probe, asm generation
and assembly does not need it, assuming proper LLVM configuration. So we provide
a comgr shim, similar to the hip shim.

`tools/setup.sh --with-stinkytofu` sets `STINKYTOFU_USE_STUB_COMGR` automatically
as appropriate for the target:

- ON on macOS (no ROCm -- the no-op stub under `fake_comgr/` satisfies
`find_package(amd_comgr)`),
- OFF on Linux (link the real ROCm comgr).

## Test

```bash
(cd build && ninja install)
stinkytofu-opt --arch gfx1250 contrib/stinkytofu/test/roundtrip-minimal.stir --print-output
lit build/contrib/stinkytofu/test -v
```

## Pin

The relevant `rocm-libraries` commit is recorded in `contrib/stinkytofu/COMMIT`
and is the single source of truth for stinkytofu.

### Regenerating the instruction tables

StinkyTofu describes each target's instruction set as hand-authored `.def`
tables (`hardware/src/gfx/Gfx<arch>/Gfx<arch>Instructions.def` and the matching
`...Formats.def`). A build-time generator, `tools/tablegen/GenInstructions.cpp`,
parses them into `.inc` C++ (`Gfx<arch>_{init,costs,operands,hwreg}.inc`) that the
StinkyTofu build then compiles. CMake runs this automatically and the `.inc`
files are not checked in. To regenerate or inspect them by hand, run the
generator from the StinkyTofu source tree
(`contrib/stinkytofu/third_party/stinkytofu/shared/stinkytofu`):

```bash
clang++ -std=c++20 -I include -I hardware/include \
  contrib/stinkytofu/third_party/stinkytofu/shared/stinkytofu/tools/tablegen/GenInstructions.cpp \
  contrib/stinkytofu/third_party/stinkytofu/shared/stinkytofu/tools/tablegen/GenInstructionsMain.cpp \
  -o /tmp/gen
/tmp/gen --arch=gfx1250 --input-dir=contrib/stinkytofu/third_party/stinkytofu/shared/stinkytofu/hardware/src/gfx --output-dir=/tmp/gfx1250-inc
```
