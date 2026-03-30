# 03: Vector Addition

First kernel that loads data from memory, computes, and stores back.
`c[tid] = a[tid] + b[tid]` with three buffer arguments, backed by numpy on CPU
and HIP for CPU <-> GPU transfers.

## Key concepts

- Three kernel arguments: `read_only` inputs (a, b) and `write_only` output (c).
  Each pointer is loaded from the kernarg segment with `s_load_dwordx2` at
  offsets 0, 8, 16.
- `func.func` for asm-level composition: the kernarg loading boilerplate is
  isolated in a reusable `@load_3_ptrs` function. The `inline` pass replaces
  the `func.call` with the function body at zero cost -- the final assembly is
  identical to having written everything inline. This is how ASTER enables
  building reusable asm-level libraries.

Note: `inline` and `symbol-dce` ensure all `func.func` functions have been inlined
and erased once they are dead. This is currently necessary as ASTER does not
yet support true function calls in the final assembly.

Note: Custom ABI conventions and composable register allocation for reusable
0-cost function calls could be implemented. Reach out if this is something of
interest.

## Run

```bash
python examples/03_vector_add/run.py                       # execute on GPU, verify with numpy
python examples/03_vector_add/run.py --print-asm           # see the assembly (func.call fully inlined)
python examples/03_vector_add/run.py --print-ir-after-all  # IR after each pass
examples/profile.sh examples/03_vector_add/run.py          # trace dumped to ./trace_xxx
```
