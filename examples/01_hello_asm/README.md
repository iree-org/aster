# 01: Hello Assembly

Simplest possible ASTER kernel. Hand-written MLIR, hand-allocated registers.

What you see in the MLIR is what you get in the assembly: no compiler passes, no
automatic optimization. This is the foundation everything else builds on.

## Key concepts

- `amdgcn.module` targets a specific GPU (gfx942 = MI300X)
- `alloca : !amdgcn.vgpr<N>` declares preallocated physical register vN and
returns an SSA value to use it in the IR
- `vop2 v_add_u32 outs %dst ins %a, %b`: preallocated registers behave like
memory locations (side-effects) and are subject to name-based RAW, WAR and WAW
considerations
- No compiler passes, direct translation to asm

Note this minimal example already exhibits an interesting control / automation
tradeoff: the kernel does not touch memory and the only observable "side-effects"
are the read/write to registers that would traditionally be considered "dead code"
by a higher-level compiler.

ASTER generally lets you write any piece of valid asm with the maximal amount of
control and can be useful for e.g. hardware validation and debugging.

## Run

```bash
python run.py
```
