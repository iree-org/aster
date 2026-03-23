# Early-Load Pipelining — Current State & TODOs

**Date:** 2026-03-22

---

## Goal

Overlap global HBM loads with MFMA compute by issuing loads for `k+K_T` immediately
after the per-iteration barrier (while the MFMA compute phase is running), targeting
700+ TFLOPS for the 16x16 fp16 kernel on MI300X.

**Expected gain:** 22–26% over the baseline (buffer-load) kernel that already reaches
522–528 TFLOPS.

---

## Implementation Overview

```
Prologue:
  global_load(k=0) → LDS
  barrier

Main loop (k = 0 .. K_TILES-2*K_T, step K_T):   ← Option C: excludes last tile
  early_global_load(k_next = k+K_T)              ← issued right after barrier
  MFMA compute(LDS[k])                           ← ~256 cycles, hides ~100-cycle L2 latency
  write LDS[k+K_T]                               ← global loads done by now
  barrier

Epilogue (k = K_TILES-K_T):
  MFMA compute(LDS[K_TILES-K_T])                 ← no early loads (no k+K_T to fetch)
```

The key insight is that one `alloc_lds/dealloc_lds` pair covers both the read and write
phases in the same iteration. The LDS allocator assigns constant offsets, so the same
byte range is reused each iteration (prologue offset = loop offset = epilogue offset).

---

## Correctness Bugs Found

### Bug 1: OOB global load on last loop iteration (FIXED by Option C)

**Symptom:** Illegal memory access crash for large multi-WG configurations.

**Root cause:** The original loop ran `k = 0 .. K_TILES`, so the last iteration issued
early loads for `k_next = K_TILES` (out of bounds). The buffer descriptor in
`make_raw_buffer_rsrc` uses `num_records=0xFFFFFFFF` (no hardware bounds check), so OOB
accesses hit real memory at a wrong-row address rather than returning 0. For large M, N
this address falls within the allocated buffer region → no page fault but corrupt data is
written to LDS.

**Fix (Option C):** Loop runs `k = 0 .. K_TILES-K_T` (one fewer iteration). Last tile
is computed in a separate epilogue block with no early loads.

### Bug 2: Wrong results for K divisible by 256 (= 4 * K_T * 32 for K_T=2)

**Symptom:** Correctness tests pass for K=64..704 except K=256, 512, 768, …
(multiples of 256). Bad elements: rows 0..63 (first 4 M-tiles) for waves with odd `wn`
(waves 1 and 3).

**Likely root cause:** Same as Bug 1. For iterations-divisible-by-4 problem sizes the
OOB last-iteration load happens to corrupt LDS data that the MFMAs still need (wrong-row
HBM data written to the LDS region used by waves with odd `wn`). Expected to be fixed by
Option C.

**Verification needed:** Run the correctness test suite after applying Option C.

---

## Assembly Observations (m4864×n4096×k8192, mt=128, nt=128, kt=2, nw=4, sw=2)

From `/tmp/asm_output.txt`:

- Loop counter: `s0` (step=2), condition: `s_cmp_lt_i32 s0, 256`.
- `v82..v97`: LDS write addresses — computed ONCE in prologue, never clobbered in loop.
- `v100..v103`: LDS read addresses — computed ONCE after prologue barrier, never clobbered.
- All 64 AGPRs (`a0..a63`) correctly zeroed in prologue.
- Resources: `vgpr=104, agpr=64, lds=32768, sgpr=19` → fits 2 WG/CU comfortably.

---

## Applied Optimizations (Context)

| # | Optimization                              | Result          |
|---|-------------------------------------------|-----------------|
| 6 | Buffer loads (MUBUF) instead of flat      | +12.5 to +24%   |
| 7 | CTA swizzle=2                             | +5.4% (M=4864)  |
| 1 | Intra-step prefetch for 32x32             | −13% (reverted) |
| 2 | Relax pipeline depth cap for 32x32        | −37% (reverted) |
| 8 | Software pipeline depth ≥ 2               | −37% (reverted) |

Current best (buffer loads + swizzle=2): **525 TFLOPS** on 4864×4096×8192.

---

## TODOs

### P0 — Fix and Validate Early-Load Kernel

- [x] **Option C: restructure loop** — loop runs K_TILES/K_T - 1 iterations; epilogue
  handles the last tile without early loads. (Implemented in this session.)
- [ ] **Run correctness tests** after Option C:
  ```
  python contrib/gemm/bench/bench_gemm_fp16_lds.py --correctness --k 256 512 768 1024
  ```
  Expect all to pass. If any fail, the root cause for Bug 2 is deeper than Bug 1.
- [ ] **Run full correctness sweep** over K = 64 .. 2048 (step 64) for K_T=2, K_T=4.
- [ ] **Run benchmark** on 4864×4096×8192 with early-load kernel and compare to baseline:
  ```
  python contrib/gemm/bench/bench_gemm_fp16_lds.py --m 4864 --n 4096 --k 8192
  ```
  Target: ≥ 650 TFLOPS (expected ~640–660 from +22–26% on 525 base).
- [ ] **Document results** in `perf_suggestions.md` section 10.

### P1 — Performance Investigations

- [ ] **Measure pipeline efficiency**: Check if the `~256 cycle MFMA compute` actually
  hides the L2 latency (~100 cycles). If not, consider if a `s_waitcnt` placement
  adjustment helps.
- [ ] **Verify A-register blocking** (perf_suggestions.md §4): confirm no spurious
  `ds_read` or `scratch_load` for A inside the n-loop. Use `--print-asm` and grep for
  `ds_read` between the first and last MFMA of a k-step.
- [ ] **Increase k_tile for 32x32** (perf_suggestions.md §3): add `k_t=8` (k_tile=256)
  to the 32x32 sweep. Expected +5–15%.
- [ ] **Swizzle=2 for 32x32** (perf_suggestions.md §5): add `swizzle=2` to the 32x32
  sweep. Cheap to try.

### P2 — Baseline Comparison

- [ ] **rocBLAS / hipBLASLt baseline** (perf_suggestions.md §9): run
  ```
  hipblas-bench -f gemm -r f16_r --transposeA N --transposeB T -m 4096 -n 4096 -k 4096
  ```
  Expected: 1100–1250 TFLOPS. Our gap is currently ~2×; understand where we stand.

### P3 — Code Quality

- [ ] **Remove unused functions** from `gemm_16x32_f16_k_loop_helpers.mlir`:
  - `@k_load_a_flat_into` / `@k_load_b_flat_into` (cross-iteration approach, abandoned).
  - `@k_fused_lds_read_compute_2d_32x32` (section 9, reverted intra-step prefetch for 32x32).
- [ ] **Update comments** in `gemm_fp16_lds.mlir` loop to remove stale reference to
  "Out-of-bounds accesses on the last iteration return 0" (no longer true: Option C
  prevents OOB accesses entirely).
- [ ] **`/export` command** (from prior session): figure out what this needs and handle it.

---

## Open Questions

1. **Bug 2 root cause:** Is the multiples-of-4 correctness failure purely caused by the
   OOB load (Bug 1), or is there a separate issue? To isolate: check if the failure
   pattern (odd `wn`, rows 0..63) correlates with the wave/tile layout affected by the
   wrong-row HBM data.

2. **L2 hit rate for early loads:** The gain estimate (22–26%) assumes L2 hits for the
   prefetched k+K_T tile. For K_T=2 and k_tile=64 bytes/row × 16 rows × K_T = 2KB of
   A-data and 2KB of B-data per wave: does this fit in the L2? Check L2 hit counters.

3. **Epilogue barrier:** After the loop (Option C), the last loop barrier guarantees all
   waves have LDS[K_TILES-K_T] ready. The epilogue reads immediately — no extra barrier
   needed. Confirm this assumption holds for 0-iteration case (K_TILES = K_T): the
   prologue barrier covers this.
