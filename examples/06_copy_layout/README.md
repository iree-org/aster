# Example 06: Layout-Based Copy with Per-Dimension OOB

`copy_2d_padded.py` is a persistent grid-stride 2D `M x N` copy.
It exercises OOB over-reads that dropped by buffer_load / store ops and
coordinate-tensor predication.

## Key concepts

- Each lane moves 16 bytes per transfer via `buffer_load_dwordx4` /
  `buffer_store_dwordx4` with the non-temporal (`nt`) modifier.
- `LogicalTensor.elem_layout` gives the dense per-dimension byte strides; the
  byte address is their dot with the recovered `(row, col)`.
- Per-dimension OOB: `CoordTensor.tiled` builds a coordinate tensor (one
  projection map per dim) tiled identically to the data, so each lane carries its
  absolute `(row, col)`. The inner-dim check `col < N` is emitted as an
  `elem_less` predicate -- never a linearized index comparison, so an interior
  over-read is the true coord `>= N`, not an aliased flat index. The outermost
  dim is the contiguous tail, clipped by `buffer_num_records_bytes`.
- The byte space is tiled with the layout algebra (`tile()`).

## Prefetch depth tuning

The loop body loads one tile at stage `0` (`with b.stage(0)`) and stores it at
stage `depth` (`with b.stage(depth)`). The SCF pipeliner reads those stage tags
and software-pipelines the loop, additionally lcm-unrolled by `depth`
(`PipelineConfig(lcm_unroll=True, unroll_factor_multiplier=depth)`), so the
steady state issues a burst of independent loads/stores per iteration. A larger
depth keeps more loads outstanding to saturate HBM bandwidth.

## Running

```
python examples/06_copy_layout/copy_2d_padded.py --n 300
python examples/06_copy_layout/copy_2d_padded.py --n 300 --depth 4 --print-asm
```
