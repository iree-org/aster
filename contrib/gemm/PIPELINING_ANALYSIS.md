# What Must Change for Pipelining to Outperform the Unpipelined Kernel

**Context:** The 16×16 MFMA GEMM with the top config (128×128 tile, k_tile=64, 2×2 wave
grid) achieves ~419 TFLOP/s (94% of ~443 TFLOP/s peak) without any software pipelining.
Two separate experiments showed pipelining is currently harmful for this config:

| Attempt | Change | Result |
|---------|--------|--------|
| Naïve outer-loop pipeline (k_tile=64, 2-stage) | LDS doubles to 64 KB → 1 WG/CU | 305 TFLOP/s (−27%) |
| LDS-aware pipeline (k_tile=32, 2-stage) | Correct occupancy preserved | 362 TFLOP/s (+5% vs k_tile=32 baseline, but −13% vs k_tile=64) |

There are two orthogonal obstacles preventing pipelining from helping.

---

## Obstacle 1 — The outer-loop pipeline is LDS-limited

### Root cause

Every pipeline stage keeps one additional copy of A+B LDS live simultaneously. The
hardware provides 64 KB of LDS per CU. Two workgroups sharing a CU each require their
LDS to fit within 64 KB total, so for 2 WGs/CU the per-WG budget is 32 KB.

For the 128×128 tile family:

```
num_stages × total_lds_bytes ≤ 32768   (to keep 2 workgroups/CU)
```

| k_tile | total_lds | Max pipeline stages (2 WG/CU) | MFMA cycles/step | HBM latency hidden by 2-WG? |
|--------|-----------|-------------------------------|------------------|-----------------------------|
| 32     | 16 KB     | 2                             | ~256 cy          | No (256 < ~300) → pipeline helps |
| 64     | 32 KB     | **1 (no pipeline)**           | ~512 cy          | Yes (512 > ~300) → pipeline hurts |
| 128    | 64 KB     | **0 (impossible)**            | ~1024 cy         | Already compute-bound |

For k_tile=64, the 512-cycle compute window already exceeds HBM latency (~300 cycles),
so natural 2-WG inter-workgroup overlap fully hides memory access. Introducing a
pipeline stage doubles LDS to 64 KB, evicting the second workgroup from the CU and
destroying the mechanism that was providing the latency hiding in the first place.

### Fix A — Accept the LDS constraint: reduce k_tile

Reducing k_tile to 32 (16 KB LDS) allows a 2-stage pipeline (32 KB) while preserving
2 WGs/CU. This is implemented in `_make_substitutions` via:

```python
max_stages_for_2wg = 32768 // cfg.total_lds_bytes
num_stages = min(4, max_stages_for_2wg, max(1, k_outer_iters))
```

Measured result: **+5.1%** for k_tile=32 (344.8 → 362.2 TFLOP/s). However k_tile=32
is slower than k_tile=64 regardless: halving k_tile doubles the outer-loop iteration
count, so more barrier and loop-counter overhead accrues relative to MFMA work. The
pipelined k_tile=32 (362 TFLOP/s) never catches the unpipelined k_tile=64 (419 TFLOP/s).

### Fix B — Use the 32×32×8 MFMA variant

`v_mfma_f32_32x32x8_f16` processes 8 K-elements per step instead of 16, doubling
k_inner for a given k_tile while LDS is unchanged. For k_tile=64, the 32×32 path gets
k_inner=8 and 2× as many MFMAs per outer step with identical LDS — a larger
compute-to-LDS ratio that creates room for both a deep pipeline and 2 WGs/CU. This is
why the 32×32 path already has 4-stage pipelining enabled and benefits from it. The
16×16 path cannot match this ratio at k_tile=64 without changing the tile geometry.

### Fix C — Static double-buffer allocation (structural kernel change)

The current kernel allocates LDS **inside** the K loop using `amdgcn.alloc_lds` /
`amdgcn.dealloc_lds`, which causes the software pipeliner to allocate separate LDS
instances for each pipeline stage. An alternative design allocates two static A/B
buffer pairs outside the loop and manually ping-pongs between them with an explicit
`buf_idx = (k / k_t) % 2` index. This makes the double-buffer cost explicit and lets
the kernel author guarantee the pipeline does not consume more than 2× LDS. However,
the LDS budget constraint is identical: for k_tile=64 the two buffer pairs still total
64 KB, which still forces 1 WG/CU. The real benefit of this approach is that it
separates the pipeline depth concern from the LDS allocator, enabling the pipeliner to
target a *sub-k_tile* prefetch granularity without creating fully independent LDS copies.

---

## Obstacle 2 — The inner-step compute stalls on LDS reads

### Root cause

This is independent of Obstacle 1 and is responsible for the remaining ~6–8% gap from
theoretical peak even in the best unpipelined configuration.

The call sequence for each outer K-step is:

```
k_read_lds_a_2d(...)              ← issues ALL 16 A ds_reads upfront
k_read_lds_b_2d(...)              ← issues ALL 16 B ds_reads upfront
k_wait_and_compute_mfmas_2d(...)  ← for each MFMA: s_waitcnt lgkmcnt(N), then MFMA
```

Both `k_read_lds_a_2d` and `k_read_lds_b_2d` loop over all (m, k_t, kh) and (n, k_t,
kh) and fire all ds_reads before the compute loop starts. `k_wait_and_compute_mfmas_2d`
then resolves each future with `get_lds_read_value_vx2` which emits a
`s_waitcnt lgkmcnt(N)` before each MFMA. With 32 outstanding reads and LDS latency of
~20–40 cycles each, the first several MFMAs must stall until earlier reads complete.

This produces exactly the staircase `s_waitcnt lgkmcnt(15)…lgkmcnt(0)` pattern visible
in the generated ASM: the first ~16 MFMAs stall with lgkmcnt(15), then each subsequent
MFMA stalls for one more read to complete.

### Fix — Fuse LDS reads into the compute loop (intra-step prefetch)

Issue the LDS reads for k_inner step `kh+1` just before the MFMA for step `kh`, rather
than all reads upfront. This means restructuring `k_wait_and_compute_mfmas_2d` (and
merging the now-unnecessary `k_read_lds_a_2d` / `k_read_lds_b_2d` pre-loops into it)
so that:

```
// Proposed structure inside k_wait_and_compute_mfmas_2d
issue ds_read A[m, kh=0], B[n, kh=0]   ← issue first step ahead of time
for kh in [0, k_inner):
    if kh+1 < k_inner:
        issue ds_read A[m, kh+1], B[n, kh+1]   ← prefetch next step
    s_waitcnt lgkmcnt(≤2)               ← at most 1 A + 1 B outstanding
    MFMA(A[m, kh], B[n, kh], C[m, n])
```

With only 2 reads in flight at a time instead of 32, the `s_waitcnt` stall shrinks from
~30 read-latency cycles to near zero for all but the very first MFMA. This fix:

- Requires **no additional LDS** — it is purely an instruction-scheduling change.
- Works for any k_tile value (including k_tile=64).
- Targets the remaining 6–8% gap from theoretical peak in the current best config.
- Requires changes to `k_read_lds_a_2d`, `k_read_lds_b_2d`, and
  `k_wait_and_compute_mfmas_2d` in
  `contrib/kittens/test/gemm_16x32_f16_k_loop_helpers.mlir`.

---

## Summary

| Fix | LDS cost | Applies to | Expected gain |
|-----|----------|------------|---------------|
| Reduce k_tile (implemented) | None | k_tile=32 only | +5.1% vs k_tile=32 baseline |
| Use 32×32 MFMA variant | None | Any tile | Enables deeper pipeline with same LDS |
| Static double-buffer | Same 2× LDS | Structural improvement | Cleaner design, same occupancy tradeoff |
| Intra-step LDS prefetch | **None** | Any config, incl. k_tile=64 | Closes ~6–8% remaining gap |

The intra-step LDS prefetch (Obstacle 2 fix) is the highest-value change: it does not
require any LDS budget change, is compatible with the existing k_tile=64 top config,
and targets the stall cycles that account for the gap between 94% and 100% of peak.
