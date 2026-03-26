# Review: `gemm_fp16_lds.mlir`

**Date:** 2026-03-25
**Reviewer:** Claude (claude-sonnet-4-6)
**File:** `contrib/gemm/gemm_fp16_lds.mlir`
**Kernel:** 16×16 `v_mfma_f32_16x16x16_f16` on gfx942 (MI300X)

---

## Status

This file implements the **Option C early-load** structure (documented in
`early_load_state.md`): global loads for `k+K_T` are issued during the current
iteration's compute phase to overlap HBM latency with MFMAs.  It is **not** the
reverted clean HEAD described in `findings.md` as the stable baseline.  Correctness
after Option C was explicitly marked unverified in `early_load_state.md`.

---

## Correctness Issues

### Bug 1: `kPone = k + 1` — wrong for K_T > 1 (line 184) ← **FIXED**

```mlir
// Before (bug):
%kPone = arith.addi %k, %c1 : index
// After (fix):
%kPone = arith.addi %k, %c_K_T : index
```

The outer loop advances `%k` by `K_T` (step `%c_K_T`), so the starting raw
K-tile index for the next outer step is `k + K_T`.  Inside `k_load_a_flat`
(helper line 624):

```mlir
%k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
```

`%k` is the base tile index.  Passing `k+1` instead of `k+K_T` loads tiles
`[k+1, k+1+K_T)` instead of the correct `[k+K_T, k+2*K_T)`.

For K_T=1 this coincidentally works (`%c1 == %c_K_T`).
For K_T=2 (k_tile=64, the best-performing config) the prefetch is one tile off
in each iteration — every outer step loads the wrong K-columns.

### Bug 2: `nIts = K_TILES - 1` — OOB prefetch for K_T > 1 (line 182) ← **FIXED**

```mlir
// Before (bug):
%nIts = arith.subi %c_K_TILES, %c1 : index
// After (fix):
%nIts = arith.subi %c_K_TILES, %c_K_T : index
```

With K_T=2 and K_TILES=256 the loop runs to k=254 and computes `kPone=k+K_T=256`.
`k_load_a_flat(k=256)` computes offsets `(256+0)*32=8192` and `(256+1)*32=8224`,
both out-of-bounds for K=8192 (tiles are indexed 0..K_TILES-1=255).

This is the exact OOB crash documented in `findings.md` Bug 1 / `early_load_state.md`
Bug 1.

With the fix (`nIts = K_TILES - K_T`):
- Loop runs k=0, K_T, …, K_TILES−2*K_T → the last kPone is `K_TILES−K_T` (in-bounds).
- Epilogue stores and computes the last outer step (tiles `[K_TILES-K_T, K_TILES)`).
- For K_T=1: `K_TILES − K_T = K_TILES − 1`, behaviour unchanged.

Both fixes together eliminate the OOB and produce the correct next-step index for
all valid K_T values.  Verification: run the correctness suite with K_T=2 and K_T=4
configs after applying the fixes.

---

## Code Quality Issues

### Issue 3: Missing `amdgcn.dealloc_lds` (lines 177–180)

The 32x32 variant calls `amdgcn.dealloc_lds` after each iteration's compute phase.
The 16x16 kernel never deallocates `%lds_a_h`/`%lds_b_h`.  If the LDS allocator
tracks live ranges for pipelining this will be a problem; at minimum the asymmetry
is confusing.

### Issue 4: Stale comment at lines 175–176

```mlir
// LDS is allocated inside the loop so the pipeline can double-buffer (depth 2).
```

`alloc_lds` is at lines 177–180, which is **before** the `scf.for` at line 183
— not inside the loop.  Pipelining is also disabled (`{{STAGE_*}} = 0` in
`_make_substitutions`).  Already flagged as a TODO in `findings.md §P4`.

### Issue 5: Pipeline stages hardcoded to 0 in `gemm_fp16_lds.py` (lines 292–297)

`_make_substitutions` computes `num_stages`/`stage_gl/dw/dr/c` but ignores them,
substituting `str(0)` for every stage constant.  The 32x32 path correctly uses the
computed values.  This is intentional (pipeliner has known compatibility issues) but
the dead computation is confusing and the `sched.rotate_head` annotations on the
barriers remain meaningful only when pipelining is active.

### Issue 6: Unsigned division for wave decomposition (lines 151–152)

```mlir
%wm = arith.divui %wid, %c_NUM_N_WAVES : index
%wn = arith.remui %wid, %c_NUM_N_WAVES : index
```

Per `CLAUDE.md`: prefer signed types.  Should be `arith.divsi`/`arith.remsi`
since both inputs are always non-negative.

---

## Performance vs. Top GEMMs on GFX942

| Kernel                         | TFLOPS (FP16, large sq.) | % of peak |
|-------------------------------|--------------------------|-----------|
| **This kernel (best)**        | ~525                     | ~40%      |
| Triton AMD (BLOCK=256, K=64)  | ~950–1050                | ~73–80%   |
| rocBLAS / hipBLASLt           | ~1100–1250               | ~84–96%   |
| MI300X peak                   | ~1307                    | 100%      |

The ~2.4× gap to rocBLAS has well-identified root causes:

### Gap 1: No overlapped global loads (~22–26% expected gain — highest priority)

The Option C structure in this file is the right shape to overlap HBM loads with
MFMAs, but was blocked by:
1. The two correctness bugs above (now fixed).
2. Software-pipeliner incompatibilities (`sched.stage` ordering conflicts,
   AGPR data layout LLVM error for ≥2 loop iterations outside `scf.for`).

Once the bugs are fixed and correctness verified, the pipeliner blockers are the
next target.  See `early_load_state.md` for full details.

### Gap 2: Software pipeline depth ≥ 2 (blocked by LDS budget)

Depth-2 requires 65 KB LDS → 1 WG/CU → measured −37%.  The bottleneck is HBM
latency (~840 cycles) vs. MFMA compute (~256 cycles); 2 WG/CU wave-level
parallelism is essential to hide it.

### Gap 3: 32×32 MFMA underperforms 16×16

Root cause: v_mfma_f32_32x32x8_f16 uses 16 AGPRs/tile vs 4 for 16×16, giving 4×
fewer output tiles per wave at the same register budget.  With fewer tiles per wave
there is less reuse per LDS load.  Untried: k_tile=256 (K_T=8) for 32×32 to raise
arithmetic intensity.  rocBLAS uses k_tile=64–128 for 32×32.

### Gap 4: CTA swizzle is problem-size dependent

swizzle=2 gives +5.4% for M=4864 but −8% for M=9728.  The main aster kittens
library uses column-major block ID traversal (N-first) which reuses A tiles across
consecutive CTAs rather than B tiles — a more robust strategy for varied problem
sizes.

### Gap 5: Direct-B path (no LDS for B) — from main repo

The top benchmark configs in `bench_perf_sweep_001_gemm_fp16_weak_scaled.py` are
tagged `direct_flat`: B is pre-shuffled (via `bpermute`) directly into the VGPR
layout matching the MFMA B-fragment, bypassing LDS entirely.  This:
- Eliminates B LDS (frees ~16 KB → fits 2 WG/CU even with A double-buffering).
- Eliminates B `ds_write_b64` + `ds_read_b64` + associated waits and barriers.
- Allows independent B global-load pipeline depth.

This is the single largest structural difference between this kernel and the top
kittens configs.

### Gap 6: LDS bank conflicts

The XOR swizzle (from `lds_16x64_b.mlir`) avoids conflicts.  Top GEMMs also
apply row padding (+8 or +16 bytes/row) as an alternative.  Bank-conflict stalls
are not currently measurable without `SQ_WAIT_INST_LDS` profiling via rocprof.

---

## Summary of Actionable Items

| Priority | Issue | Status |
|----------|-------|--------|
| P0 | `kPone = k + 1` wrong for K_T > 1 (line 184) | **Fixed** |
| P0 | `nIts = K_TILES - 1` causes OOB for K_T > 1 (line 182) | **Fixed** |
| P0 | Run correctness suite with K_T=2,4 cases after fixes | Pending |
| P1 | Add `dealloc_lds` for symmetry with 32x32 | Pending |
| P1 | Fix stale LDS-in-loop comment (lines 175–176) | Pending |
| P1 | Investigate pipeliner stage-ordering conflict | See `early_load_state.md` |
| P2 | `arith.divui`/`remui` → `arith.divsi`/`remsi` (lines 151–152) | Pending |
| Perf | Early-load pipelining blockers | See `early_load_state.md` |
| Perf | k_tile=256 sweep for 32×32 | See `perf_suggestions.md §3` |
| Perf | Direct-B path (no LDS for B) | See main repo kittens library |
| Perf | rocBLAS baseline measurement | See `perf_suggestions.md §9` |
