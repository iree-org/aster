# Example 07: Minimal Pin-Fusion

This example is a small, focused version of fused `add + relu` with a mixed
allocated/unallocated handoff.

## Shape

- `add4(a_ptr, b_ptr) -> v40..v43`
  Loads two `dwordx4` payloads into pinned `v12..v19`, then adds into pinned
  `v40..v43`.
- `bridge_add_to_relu4(v40..v43) -> 4 x !amdgcn.vgpr`
  Converts pinned outputs to unallocated VGPR values.
- `relu4(x0..x3, out_ptr)`
  Computes `max(x, 0)` and writes with `global_store_dwordx4`.

Main files:

- `examples/07_simple_fusion/simple_fusion.py`
- `examples/07_simple_fusion/add4.mlir`
- `examples/07_simple_fusion/pin-fusion-asm.mlir`

## Run

```bash
python examples/07_simple_fusion/simple_fusion.py --print-ir-after-all
python examples/07_simple_fusion/simple_fusion.py --print-asm
python examples/07_simple_fusion/simple_fusion.py
```

Use `pin-fusion-asm.mlir` when you want a compact IR/ASM-check fixture for the
same structure.
