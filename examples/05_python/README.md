# 05: Python Meta-Programming

Same `out[tid] = in[tid] + 42` as example 04, but the MLIR is generated
from Python via `KernelBuilder` instead of written in a `.mlir` file.

Additional layers of syntactic sugar and DSL features can be layered on top if /
when useful.

## Key concepts

- `KernelBuilder("add42", "kernel", target="gfx942", isa="cdna3")`: creates
  a builder that emits AMDGCN MLIR ops into a fresh module.
- `b.add_ptr_arg(AccessKind.ReadOnly)` + `b.load_args()`: declare and load
  kernel buffer arguments. Replaces `amdgcn.load_arg`.
- `b.thread_id("x")`, `b.affine_apply(...)`, `b.global_addr(...)`: index
  computation. Python expressions generate MLIR affine maps.
- `b.global_load_dword(addr)`, `b.v_add_u32(a, b)`,
  `b.global_store_dword(val, addr)`: emit GPU instructions.
- `b.wait_vmcnt(0)`: explicit wait (KernelBuilder uses the synchronous model).
- `b.build()`: finalize and return the MLIR module.
- `compile_module(module, ...)`: compile the programmatic module to assembly
  (same pipeline as file-based compilation).

Note: No `.mlir` file in this directory -- the kernel lives entirely in `run.py`.

## Run

```bash
python examples/05_python/run.py                       # execute on GPU
python examples/05_python/run.py --print-asm           # see the assembly
python examples/05_python/run.py --print-ir-after-all  # see generated MLIR + passes
examples/profile.sh examples/05_python/run.py          # trace dumped to ./trace_xxx
```
