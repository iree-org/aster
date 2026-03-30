# 02: NOP Insertion

AMD GPUs execute memory operations asynchronously. When a `global_store_dword`
is in flight reading v0 and the next instruction overwrites v0, the hardware
would read stale data. The `amdgcn-hazards` pass analyzes the instruction
stream and inserts `v_nop` delays where needed.

Note: proper insertion `v_nop` and `s_nop` instructions is notoriously difficult
by hand, and is often wrong in most AMDGPU kernels. This is an immediate area
where a little compiler automation goes a long way without sacrificing control.

## Key concepts

- More instruction types and classes.
- Kernel arguments and types with an ABI convention automatically populated upon.
  kernel launch. For now the convention is implicit but potentially error-prone
  (i.e. the output buffer pointer is "known to" reside in `s[0:1]`).
- `global_store_dword v1, v0, s[2:3]`: store v0 to memory at base s[2:3] +
  offset v1. The memory unit reads v0 asynchronously after dispatch.
- `v_nop`: no-operation that burns one cycle, giving the memory unit time to
  finish reading registers before they are overwritten.
- Minimal compiler pipeline with a pass that does something interesting:
  `amdgcn-hazards` detects read-after-write hazards across instruction classes
  (VMEM, VALU, MFMA) and inserts the minimum required NOPs.
- Each thread writes its thread ID to `output[tid]`, verified by numpy on GPU.

Note: This is the first example with a kernel argument (an output buffer) and
observable memory side-effects. The self-check traps if v0 is not properly
overwritten to 7 after the store.

Note: ASTER exposes `amdgcn.load` and `amdgcn.store` operations that are not 1-1
with the ISA, although they map unambiguously to the expected counterpart. The
design goals of an IR (e.g. semantic usefulness and uniformity for programming)
are different from the design goals of an ISA (e.g. opcode considerations and
similarity for different instruction classes).

Note: At this level, we have 2 implicit conventions with `threadIdx.x` residing
in `v0` and the output pointer residing in `s[0:1]`. As we further raise the
level of abstraction in future examples, this convention will be carried by ops
and their lowerings.

## Run

```bash
python run.py                 # execute on GPU
python run.py --print-asm     # see the inserted NOPs
```
