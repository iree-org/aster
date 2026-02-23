# ASTER 💫 : Assembly Tooling and Representations

MLIR C++ tool for programmable and highly-controllable assembly production on
AMD GPUs.

## Motivation

Today, achieving peak performance on modern AI accelerators often requires control
over low-level hardware features. This trend is expected to further exacerbate as
more asynchronicity and dynamism are built first-class in the hardware. As Dark
Silicon trends continue, hardware is expected to expose coarser-grain primitives
and coarser-grain programming models must be used (e.g. with warp/wave
specialization, the low-level programming model increasingly resembles
MPI/MIMD-style parallelism but complexified by low-level hardware constraints
such as instruction issue ports or warp/wave scheduling and specialization).

AMD's open approach to hardware ISA documentation creates a unique opportunity to
build world-class assembly tooling in the open, making AMDGPU ASM accessible to a
broader community as well as higher-level tools, while maintaining expert-level
control.

To reap the benefits of modern and future HW we believe an order of magnitude
better low-level tooling is needed.

Aster builds the foundations for highly-controllable assembly production
and pushes the boundaries of what’s possible in low‑level performance tooling.

## Design Philosophy and More

**ASTER** is an open-source MLIR-based tool for programmable and
highly-controllable assembly production on AMD GPUs.
We believe ASTER addresses a gap in the ML high-performance software stack:
the ability to write, optimize, and reason about the lowest level of hardware
with the same rigor, composability, type-safety and automation that are available
at higher levels in the software stack.

ASTER embraces three core principles:

### 1. **Control First, Automatic Optimizations Second**
### 2. **Leverages Modern Compiler Infrastructure**
### 3. **Composable by Design**

For a quick TL;DR, look at our little demo, [here](demo/README.md).
Read more about the ASTER design principles, [here](#design-philosophy).

## Disclaimer:

ASTER is a very young project that we are pre-releasing early to get exposure and
user feedback while collaborating in the open. At this time, do not expect
stability of production readiness.

## Supported GPU targets

ASTER has early .hsaco generation support for the following targets, which all
require an appropriate LLVM AMDGPU backend for translating asm to binary:

| Target   | ISA   | Product Family          |
|----------|-------|-------------------------|
| gfx940   | CDNA3 | MI300A                  |
| gfx942   | CDNA3 | MI300X                  |
| gfx950   | CDNA4 | MI350X                  |
| gfx1201  | RDNA4 | Radeon RX 9070          |

HSACO binary generation requires an LLVM build that recognizes the target chip.
Newer targets may require a recent LLVM version.
ASTER's own IR and ASM translation works regardless of LLVM version.

Support for newer architectures is very preliminary and subject of active ongoing
work.

## Building and Testing ASTER

### Quick start

```bash
tools/setup.sh
```

This handles everything: prerequisites check, shared LLVM build, venv, cmake,
and build. Run `tools/setup.sh --help` for options. See below for manual steps.

### Manual setup

ASTER can be built on macOS and Linux. Windows is also expected to work
but is less tested at this time. Examples and tests are meant to always
cross-compile and build valid HSACO on any host machine.
Integration tests that require execution on actual hardware are filtered with
appropriate pytest and lit filters.

Generally, once the first LLVM compilation occurred, we aim at keeping builds and
tests always running within a (parallel) budget of a few seconds.


### Common setup

#### venv

We use `uv` to pip install in the new venv, the latency of pure `pip` being too high.

```bash
# Create a virtual environment
python3 -m venv --prompt aster .aster

# Activate the virtual environment
source .aster/bin/activate

# Install the requirements of the project. This installs cmake, ninja...
uv pip install -r requirements.txt
```

#### LLVM requirement

ASTER requires a pre-built LLVM installation with MLIR and LLD.
See [README_devs.md](README_devs.md) for instructions on building a shared LLVM
installation.

#### Set useful variables in a Python virtual environment

Set environment variables to load automatically upon venv activation:

```bash
cat >> .aster/bin/activate << 'EOF'

export PATH=${PWD}/.aster/bin/:$(python -c "import sysconfig; print(sysconfig.get_paths()['scripts'])"):${PATH}

export PYTHONPATH=${PYTHONPATH}:${PWD}/.aster/python_packages/:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

export LLVM_INSTALL=${HOME}/shared-llvm
export CMAKE_PREFIX_PATH=${LLVM_INSTALL}:${CMAKE_PREFIX_PATH}
EOF

# For good measure, on the first instance do a:
deactivate
source .aster/bin/activate
```

#### Building (macOS and Linux without GPU)

This builds ASTER for cross-compilation only (no HIP runtime, no on-device execution).

```bash
(
  mkdir -p build \
  && cd build \
  && cmake ../ -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="../.aster" \
    -DLLVM_EXTERNAL_LIT=${VIRTUAL_ENV}/bin/lit \
  && ninja install
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

Add ROCm paths to your venv activation:

```bash
cat >> .aster/bin/activate << 'EOF'

export PATH=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/bin/:${PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/_rocm_sdk_devel/lib
EOF

deactivate
source .aster/bin/activate
```

Then build with HIP support:

```bash
(
  mkdir -p build \
  && cd build \
  && cmake ../ -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="../.aster" \
    -DLLVM_EXTERNAL_LIT=${VIRTUAL_ENV}/bin/lit \
    -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)/hip" \
    -DHIP_PLATFORM=amd \
  && ninja install
)
```

### Testing

#### Executing all tests
```
(cd build && ninja install) && lit build/test -v && pytest -n 16 ./test ./mlir_kernels ./contrib ./python
```

#### Executing lit tests only
```bash
(cd build && ninja install) && lit build/test -v
```

#### Executing pytests only
```
(cd build && ninja install) && pytest -n 16 ./test ./mlir_kernels ./contrib ./python
```

#### Running Python manually
Additionally, to run Python scripts in the absence of Python wheels for this
project, the libASTER.dylib/.so library is automatically installed alongside the
Python modules so they can find it (macOS and Linux).

The following should now properly print valid IR:

```bash
python test/python/smoke.py
```

#### Generating HSACO files

To generate HSACO files from MLIR modules, use the `assemble_to_hsaco` utility
function:

```python
from aster import ir, utils
from aster.dialects import amdgcn

with ir.Context() as ctx, ir.Location.unknown():
    # ... build your module ...

    # Translate to assembly
    asm = utils.translate_module(amdgcn_mod)

    # Assemble to HSACO (requires LLVM_TOOLS_DIR to be set)
    hsaco_path = utils.assemble_to_hsaco(asm, target="gfx942")
    print(f"Generated HSACO: {hsaco_path}")
```


## Design Philosophy

ASTER embraces three core principles:

### 1. **Control First, Automatic Optimizations Second**
-  Modifying assembly comes from a specific intent to make hardware perform
specified operations at specified times predictably; ASTER respects that intent,
while still providing opt-in composable automation to increase velocity
- WYSIWYG: What you see is what you get, no automatic compiler transformation by
default. This lets you actually run exactly the ASM you want with e.g. dead-code
elimination interference that is hard to avoid in compiler pipelines.

These features allow an unprecedented level of control which enables separation
of concerns and first-principle thinking.
As an early motivating example, we wanted to deeply understand the performance of
low-level schedules to saturate the MATRIX unit using only LDS loads + MFMA in a
critical code region. In particular, we wanted to avoid worrying about swizzle
patterns that are necessary to avoid performance bugs due to shared memory bank
conflicts.
We ended up with a simple solution: just load from LDS at address 0, in a first
approximation.
Such an experiment would immediately turn into dead code when passed through a
compiler. Instead we were able to focus on isolating ASM and hardware performance
characteristics to deeply understand the hardware characteristics.

### 2. **Leverages Modern Compiler Infrastructure**
- A clean MLIR dialect for AMDGPU instruction classes (e.g. VOPx, MUBUF, DS, etc)
- Semantic grouping for readability: one MLIR operation can represent a
family of similar instructions, syntactic improvements are introduced to increase
programmability and understanding
- SSA-based representation of ASM dialect instructions enables modern compiler
analyses

### 3. **Composable by Design**
- Minimal and independent counterpart to top-down MLIR compiler design and
abstractions, usable independently of any framework or compiler.
- Interacts with the existing MLIR ecosystem and will plug into existing compilers.
- Partial register allocation: fully specify ASM and registers of regions with
critical performance, automate the rest
- Incremental compilation: inline hand-optimized functions into target programs
- Python-first API for metaprogramming backed by C++/MLIR tooling and automated
compiler passes where useful

## Core ASTER Infrastructure Available Today

**IR Design**: Clean abstraction MLIR Dialect with DPS ops to represent AMDGCN ISA.
The following MLIR snippet:
```mlir

    // ds_load from ldsA
    %loaded_a_from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %a_reg_range, %thread_offset_f16, offset = 0
      : !amdgcn.vgpr -> !amdgcn.vgpr_range<[? + 2]>

    // ds_load from ldsB
    %loaded_b_from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %b_reg_range, %thread_offset_f16, offset = 512
      : !amdgcn.vgpr -> !amdgcn.vgpr_range<[? + 2]>

    // s_waitcnt(lgkmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // mfma - A and B need 2 VGPRs each, C needs 4 VGPRs
    %c_mfma_result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
      %c_reg_range, %loaded_a_from_lds, %loaded_b_from_lds, %c_reg_range
        : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>

    // global_store of c_mfma_result
    %c4 = arith.constant 4 : i32 // shift left by dwordx4 size (16 == 2 << 4).
    %thread_offset_f32 = amdgcn.vop2.vop2 #amdgcn.inst<v_lshlrev_b32_e32> %offset_a, %c4, %threadidx_x
      : (!amdgcn.vgpr, i32, !amdgcn.vgpr<0>) -> !amdgcn.vgpr
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx4> %c_mfma_result, %c_ptr[%thread_offset_f32]
      : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr]

    // s_waitcnt(vmcnt(0))
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
```

simply converts to the following ASM snippet (after register allocation and
without scheduling changes):

```
ds_read_b64 v[2:3], v1
ds_read_b64 v[4:5], v1 offset:512
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x16_f16 v[8:11], v[2:3], v[4:5], v[8:11]
v_lshlrev_b32_e32 v1, 4, v0
global_store_dwordx4 v1, v[8:11], s[6:7]
s_waitcnt vmcnt(0)
```

**Simple Custom Instruction Scheduling Infrastructure and Register Allocation**:
We provide a few automatic transformations that we have useful for productivity
while still preserving user control:
(a) we do incremental register allocation around user-specified register selection
if needed
(b) we support features such as multi-dimensional unrolling for complex
instruction interleaving, SSA-aware dependency tracking and liveness analysis with
SSA-based interference graphs
(c) as a consequence, one can both metaprogram the IR in Python or write compiler
passes to control advanced transformations

**Assembly Generation**:
- Direct translation from MLIR to AMDGCN assembly
- Capable of generating linked HSACO binaries without any external tooling
- Complete metadata generation (.rodata, .amdhsa_kernel directives)
- Ready-to-execute HSACO generation
- Compatible with CDNA and RDNA versions (early proof of concepts on RDNA,
current focus on CDNA)

**Minimal Dependency Python Runtime Integration**:
- NumPy-compatible kernel invocation
- HIP runtime integration
- Fast time to insight: edit → compile → execute → profile in seconds
- Hardware performance counters (s_memtime) for cycle-accurate profiling

**Testing Infrastructure**:
- FileCheck-based lit tests for IR transformations
- Integration tests with execution on real HW
- Correctness verification against NumPy reference
- CI-ready test harness


## Key Upcoming Infrastructure and Evolutions We Are Excited About
- Lowerings from higher level MLIR dialects and connection to ASTER
- Raising from existing ASM to MLIR ASTER to edit and fine-tune existing ASM
using familiar MLIR tooling, automation and type checkers
- Creation of high-performance, reusable and precisely fine-tuned nanokernels.
These will be the target of MLIR rewrite patterns from higher levels in the stack
- Heuristics for controllable automation of transformations, selectively
augmenting WYSIWYG with productivity tools
- MLIR-based IR sketches for high-performance patterns with autoscheduling of
instructions
- Connection to higher-level, end-to-end ML compilation flows, in particular
[IREE](https://github.com/iree-org/iree). Others will follow in time.

## Why We Believe This Matters for the Community
- **Insight over guesswork**: Understand exactly what your hardware is executing
- **Fast iteration**: Compile times measured in seconds, not minutes
- **Composability**: Mix hand-optimized kernels with generated code from other
sources
- **Rapid kernel development**: Test new ML kernel implementations directly on
the relevant hardware features.
- **Hardware exploration**: Deep understanding through systematic measurement
- **Reproducible results**: IR-based artifacts will be version-controllable and shareable
- **Interoperability**: MLIR ecosystem means your tools automatically compose
- **Level the playing field**: Open alternative to proprietary CUDA assembly tools
- **Community building**: Shared infrastructure reduces duplicated effort
- **Accelerate innovation**: Lower barriers to AMDGPU programming at peak performance

## Other Hardware Than AMDGPUs
While our immediate focus is to get ASTER to be good and useful for AMDGPUs,
support for more diverse classes of hardware is also something we would love to have.
As the project evolves and matures, we expect to also support other ISAs.
In the fullness of time, we favor abstractions that will compose with different
targets, in the same spirit as [MLIR](https://mlir.llvm.org/) and
[IREE](https://github.com/iree-org/iree).
If ASTER sounds appealing to your particular ISA, please reach out!


## Non Goals
- **Not a general-purpose LLVM replacement**: ASTER focuses on critical
performance regions for dense linear algebra and similar workloads that programmers
care about enough to go deep into hardware details. This is not a general-purpose
application compilation and does not support a C language abstraction or dusty
deck code.
For the vast majority of code, higher-level productivity languages layered on top
of existing LLVM pipelines remain the go-to solution.

- **Not comprehensive ISA coverage**: Instruction coverage is driven by real use
cases. ASTER aims to support the essential instructions needed for
high-value high-performance kernels, not every instruction variant in the
AMDGPU ISA. Coverage will grow organically based on needs.

- **Not fully-automatic optimization**: ASTER provides building blocks and
composable automation, but does not aim to be a "smart compiler" that
automatically achieves peak performance without expert guidance.
Control and predictability are prioritized over automatic optimization.
We expect ASTER will be a useful building block for such compilers, for AMDGPUs.

- **Not Production-Ready**: Our goal is to release early, often and get
feedback from the community in the open. While functional for early adopters,
ASTER is young and in active development since October 2025.
Error messages, debugging support, documentation, and tooling ergonomics will
continue improving, depending on community needs and feedback.
Setting realistic expectations around maturity is important.

## Early Technical Achievements

In the span of a few weeks, ASTER already demonstrated a few non-trivial
capabilities that typically take traditional compiler projects longer to achieve:

1. **Programmable AMDGPU With End-to-end Compilation** from Python to executable
HSACO binaries and connection to NumPy + HIP APIs.
1. **SSA-based register allocation** with interference analysis that composes
with partial user specifications and function inlining at the MLIR level
2. **Instruction scheduling** using simple modular expressions for precise
control over emission timing
3. **Hardware validation** with passing integration tests on real AMD GPUs
5. **Clean separation of concerns**: Authoring → Analysis → Transformation → ASM → Execution


## Conclusion

We believe assembly-level programming and introspection is not the end of the
road but that it has been waiting for the right tools. We believe ASTER may
represent a paradigm shift: bringing modern compiler engineering principles to
the lowest level of the software stack and making the hard things possible and
the complex things manageable.

**Join us in making assembly first-class.**

---

## Appendix: Quick Start

```python
from aster import ir
from aster.dialects import api

with ir.Context() as ctx, ir.Location.unknown():
    # Build IR using Python
    module = api.create_kernel(
        "my_kernel",
        args=[("input", f32_ptr), ("output", f32_ptr)],
        num_vgprs=32,
        num_sgprs=16
    )

    # Translate to assembly
    asm = utils.translate_module(module)

    # Compile to HSACO
    hsaco = utils.assemble_to_hsaco(asm, target="gfx942")

    # Execute on GPU
    result = utils.launch_kernel(hsaco, "my_kernel", inputs=[a, b])
```

## Acknowledgements

We had productive and very helpful discussions with Sasha Lopoukhine during early
experimentations based on his xDSL work.

Seb Vince's feedback was instrumental in debugging rocprof traces during the
early bringup phase.

We are also grateful for early and ongoing discussions with Alex Zinenko and his
expert feedback on this RFC.

## References

- MLIR: https://mlir.llvm.org
- AMD CDNA4 ISA: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf


---

**Repository**: https://github.com/iree-org/aster

**License**: Apache-2.0 WITH LLVM-exception

**Contact**: Nico.Vasilache@amd.com, Fabian.Mora-Cordero@amd.com
