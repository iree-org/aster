# 04: Register Allocation

Same load-add-store pattern as 03, but with virtual registers. No hand-allocated
register numbers, a graph-coloring based register allocation pass assigns `v0`,
`v1`, `s[0:1]`, etc. automatically.

## Key concepts

- `amdgcn.alloca : !amdgcn.vgpr` (no number): virtual register. The compiler
  decides which physical `vN` to use based on liveness and interference. The
  default algorithm greedily tries to reuse register.
- `amdgcn.alloca : !amdgcn.vgpr<11>`: pre-allocated (pinned to `v11`). The
  allocator treats this as a constraint and assigns virtual registers to slots
  that don't conflict. Virtual and physical registers mix freely in the same
  kernel; use physical register when you need exact control (e.g.
  hardware-mandated register positions, specific optimizations or reuse patterns),
  virtual everywhere else.
- `amdgcn.load_arg N`: loads kernel argument N into a virtual SGPR pair.
  Replaces the manual `s_load_dwordx2` + `make_register_range` boilerplate.
- `amdgcn.load ... -> !amdgcn.read_token<flat>`: memory operations return
  tokens representing in-flight async work. Tokens are SSA values -- the
  compiler tracks which loads/stores are outstanding.
- `amdgcn.wait deps %tok`: waits for a specific token to complete. The compiler
  lowers this to the minimal `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` based on
  dataflow analysis. Compare to examples 01-03 where we manually wrote
  `s_waitcnt vmcnt = 0` (wait for ALL outstanding ops). Token-based waits are
  more precise: the compiler can overlap independent loads by waiting only for
  the token it actually needs.
- `amdgcn-backend`: the pass pipeline that runs bufferization, register coloring
  (graph-coloring allocator), wait lowering, and CF legalization.
- `vop2 ... outs %dst ins %a, %b -> !amdgcn.vgpr`: DPS returns a value in SSA
  form. The compiler tracks dataflow and reuses registers when lifetimes don't
  overlap (e.g. `v0` holds both the loaded data and the result).

This is the bridge from "programmer controls every register" (examples 01-03) to
"programmer writes logic, compiler maps to hardware". This composes with
preallocated registers where needed.

## Run

```bash
python examples/04_regalloc/run.py                       # execute on GPU
python examples/04_regalloc/run.py --print-asm           # see allocated registers
python examples/04_regalloc/run.py --print-ir-after-all  # IR after each pass
examples/profile.sh examples/04_regalloc/run.py          # trace dumped to ./trace_xxx
```
