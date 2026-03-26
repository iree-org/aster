# GEMM FP16 LDS — Performance Gap Analysis

**Kernel:** `contrib/gemm/gemm_fp16_lds.mlir`
**Target:** MI300X (gfx942, CDNA3)
**Observed:** ~460 TFLOP/s
**Top GEMMs:** ~900–1000 TFLOP/s (rocBLAS, CK, Triton)
**Gap:** ~2×

---

## Ranked Root Causes

### Rank 1 — Single LDS buffer + two barriers: no compute/load overlap

**Estimated impact: ~1.4–1.8×**

The LDS buffers are allocated once, outside the K-loop (`alloc_lds` at lines 176–179).
This forces a strictly serial schedule per K-outer step:

```
LDS-store(k)  →  s_waitcnt  →  barrier₁  →  MFMA(k)  →  barrier₂  →  next iteration
```

`barrier₂` (line 209) forces **all waves to finish computing** before the next LDS store
can begin. No overlap of MFMA with the next tile's global-load or LDS-write stage is
possible.

The code is self-aware: `_make_substitutions()` (gemm_fp16_lds.py:252–259) explicitly
keeps all `sched.stage` annotations at 0 with the comment:

> "sched.stage pipelining beyond a single stage would create DS_WRITE/DS_READ races
> on the same LDS buffer. Keep all stage annotations at 0."

Top GEMMs ping-pong between two LDS buffers to eliminate `barrier₂` and run:

```
MFMA on buffer A  ∥  global-load + LDS-store into buffer B
```

---

### Rank 2 — Compute density per K-outer too low to hide the prefetched global load

**Estimated impact: ~1.3–1.5×**

Global loads for K-step `k+1` are prefetched via `iter_args` (lines 187–195). For the
load to complete before it is needed, the current MFMA phase must last ≥ 400 cycles
(MI300X HBM latency). The requirement is:

```
tiles_per_wave × k_inner × 8 cycles ≥ 400 cycles
→ tiles_per_wave × k_inner ≥ 50
```

For a representative sweep config (4 waves, `m_tile=64, n_tile=64, k_tile=32`):

| Parameter | Value |
|-----------|-------|
| `tiles_per_wave` | 4 |
| `k_inner` | 2 |
| MFMA cycles | 8 |
| Compute cycles per K-outer | **64** |
| Required | **400** |

The MFMA phase is 6× too short. The prefetch does not finish in time, and the next
iteration stalls on `s_waitcnt vmcnt(0)` inside `k_store_a_flat`/`k_store_b_flat`.

This is the reason the kernel needs high multi-WG occupancy for latency hiding — but
high occupancy requires small tiles, and small tiles cause low compute density. The two
constraints are mutually reinforcing.

---

### Rank 3 — `ds_write_b64 × 2` instead of `ds_write_b128`

**Estimated impact: ~10–15%**

`write_vx4_to_lds_at` (lds_16x64_b.mlir:72–84) splits each 16-byte `dwordx4` into two
`!vx2` halves and issues **two** `ds_write_b64` instructions per thread per tile.
A single `ds_write_b128` would:

- Halve the number of LDS write instructions
- Reduce instruction-issue pressure during the store phase
- Allow tighter scheduling of the load→store stream

---

### Rank 4 — `v_mfma_f32_16x16x16_f16` instead of `v_mfma_f32_32x32x8_f16`

**Estimated impact: ~10–20%**

Both variants have the same FLOP/cycle throughput. The difference is instruction count
for a given output tile size:

| MFMA variant | FLOPs/instruction | Latency | K-steps per K=64 | MFMAs for 64×64 output |
|---|---|---|---|---|
| 16×16×16 | 8 192 | 8 cycles | 4 | 16 × 4 = **64** |
| 32×32×8  | 16 384 | 16 cycles | 8 | 4 × 8 = **32** |

The 32×32 variant issues 2× fewer MFMA instructions for the same output tile, reducing
instruction-fetch overhead and AGPRr hazard management. The 32×32 path already exists
(`gemm_fp16_lds_32x32.mlir`) but shares the same single-buffer structure.

---

### Rank 5 — CTA swizzle not tuned to MI300X XCD topology

**Estimated impact: ~5–15% at memory-bound problem sizes**

`swizzle` defaults to 1 (row-major dispatch, gemm_fp16_lds.py:63). MI300X has 8 XCDs
of ~28 CUs each sharing an L2 cache. Without swizzle, consecutive block IDs span
multiple N-blocks, preventing B-tile reuse within an XCD's shared L2.

A swizzle of 8–16 aligns consecutive flat block IDs to the same B-column, keeping the
B tile hot in the XCD-local L2 across the ~28 CUs that process different M-blocks.
The CTA swizzle logic is already implemented (gemm_fp16_lds.mlir:133–143); it needs
tuning, not new code.

---

### Rank 6 — LDS read latency under-hidden inside `k_fused_lds_read_compute_2d`

**Estimated impact: ~5–10%**

`ds_read_b64` has ~40-cycle latency. With only 2 MFMAs per K-inner step (16 cycles),
the read issued at step `ki` is not fully hidden before its result is needed by the MFMA
at step `ki`. The comment on lines 202–204 notes that reads for `ki+1` are issued before
resolving `ki` ("intra-step prefetch"), but 1-step prefetch depth is insufficient for
40 cycles. Top kernels buffer 2–4 future K-inner steps in VGPRs.

---

### Rank 7 — Startup `s_waitcnt lgkmcnt=0`

**Estimated impact: negligible**

Line 110 issues a conservative scalar-memory fence before arg pointer preparation. It is
not on the critical path for any reasonable K, but is unnecessary.

---

## Summary Table

| Rank | Issue | Impact |
|------|-------|--------|
| 1 | Single LDS buffer + two barriers — no compute/load overlap | ~1.4–1.8× |
| 2 | Compute per K-outer insufficient to hide the prefetched global loads | ~1.3–1.5× |
| 3 | `ds_write_b64 × 2` instead of `ds_write_b128` | ~10–15% |
| 4 | 16×16 MFMA — 2× instruction overhead vs 32×32 for same output tile | ~10–20% |
| 5 | CTA swizzle not tuned to XCD topology (default = 1) | ~5–15% |
| 6 | LDS read prefetch depth of 1 insufficient to hide 40-cycle DS latency | ~5–10% |
| 7 | Startup `s_waitcnt lgkmcnt=0` | negligible |

Ranks 1 and 2 are strongly coupled. Fixing #1 (double-buffering) is what allows the
large output tiles that fix #2, and together they account for nearly the entire 2× gap.

---

## The Regime Shift Required

The current kernel operates in the **occupancy-based latency-hiding** regime:

- Small tiles → high occupancy (many WGs/CU)
- Many WGs context-switch to hide global load latency
- But small tiles → low arithmetic intensity → global load latency is still exposed

Top GEMMs operate in the **ILP-based latency-hiding** regime:

- Large tiles (e.g., 128×128 per WG) + double-buffering
- Only 1–2 WGs per CU (LDS budget consumed by two large buffers)
- The single WG's instruction stream is dense enough (`compute_cyc ≥ 400`) to hide
  loads via ILP without needing other WGs to context-switch

For a 128×128 tile, 4 waves, k_tile=64:

```
tiles_per_wave = (8×8)/4 = 16
k_inner        = 64/16   =  4
compute_cyc    = 16 × 4 × 8 = 512 cycles  ≥  400  ✓
```

With double-buffering, LDS = 2 × (8+8) × 1 KB × k_t = 32–64 KB (1 WG/CU), but loads
are hidden by ILP, not occupancy. This is the correct trade-off.
