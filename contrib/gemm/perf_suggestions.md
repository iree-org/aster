# GEMM fp16 LDS — Performance Suggestions

Target: `v_mfma_f32_16x16x16_f16` (16x16) and `v_mfma_f32_32x32x8_f16` (32x32) on gfx942 (MI300X).

---

## Why 16x16 outperforms 32x32

**Root cause: K-steps per MFMA instruction and LDS read pressure.**

| Instruction              | K per step | AGPRs/tile | ops/inst |
|--------------------------|-----------|-----------|---------|
| v_mfma_f32_16x16x16_f16  | 16        | 4         | 8192    |
| v_mfma_f32_32x32x8_f16   | 8         | 16        | 16384   |

For a fixed `k_tile=32` and a 32×32 output area, 16x16 needs 2 K-steps (`k_inner=2`)
issuing 4 LDS reads; 32x32 needs 4 K-steps (`k_inner=4`) issuing 8 LDS reads — **2×
more LDS traffic for the same compute**. CDNA3 `ds_read_b64` latency is ~20 cycles, so
this directly hurts utilisation.

**AGPR pressure reduces effective tile size.** For the same AGPR budget (e.g. 64 AGPRs):
- 16x16: 64 / 4 = **16 output tiles** per wave → high arithmetic intensity.
- 32x32: 64 / 16 = **4 output tiles** per wave → 4× less reuse per LDS load.

**Missing intra-step prefetch (Change 1).** The 16x16 kernel has
`@k_fused_lds_read_compute_2d` which issues LDS reads for step `ki+1` before waiting
for `ki`. This keeps only `m_t_per_wave + n_t_per_wave` reads in flight at each
`s_waitcnt` instead of `k_inner × (m_t_per_wave + n_t_per_wave)`. The 32x32 kernel
still uses the old separate-read-then-compute pattern.

**Larger tile geometry forces 1-stage pipeline.** The sweep formula
`m_tile = mult × num_waves × mfma_tile` produces 32x32 tiles that are 2× larger in
each dimension than the corresponding 16x16 tile for the same multiplier. The pipeline
cap `32768 // total_lds_bytes` therefore reaches 1 (no pipelining) for 32x32 configs
that 16x16 would still pipeline at depth 2.

---

## Suggestions

### 1. Port intra-step prefetch to 32x32 ❌ reverted — hurts 32x32

The 16x16 fused pattern (`@k_fused_lds_read_compute_2d`) was implemented and tested
for 32x32 but caused a ~13% regression (470 → 410 TFLOPS on the top config).

**Root cause:** the upfront-all-reads approach is already latency-optimal for 32x32
because `k_inner=4` (vs 2 for 16x16) means `k_inner × (m + n)` reads are issued
before the first wait. For typical configs (e.g. `m_t_per_wave=2, n_t_per_wave=2,
k_inner=4`): 16 reads are issued in ~32 cycles; LDS latency is ~20 cycles; all reads
complete before the first `s_waitcnt` fires — **zero stall**, for free.

The intra-step prefetch limits in-flight reads to `m + n` reads per step. With only
`m + n = 4` reads issued before the first wait (~8 cycles elapsed), the first read
(issued at T=0, ready at T=20) stalls for **~12 cycles per ki step × 4 steps = ~48
wasted cycles per K_TILE**. Added to the per-step copy overhead (cur←next), this
consistently regresses every config.

**Why 16x16 benefits but 32x32 does not:**
- 16x16 `k_inner=2`: upfront issues `2×(m+n)` reads in ~8 cycles before the first
  wait. For small tiles, reads are only ~half-latency done → stall ~12 cycles.
  The fused version runs enough MFMAs to fully hide this.
- 32x32 `k_inner=4`: upfront issues `4×(m+n)` reads — twice as many, in ~32 cycles
  — which already saturates the LDS pipeline and hides latency without any prefetch.

The `@k_fused_lds_read_compute_2d_32x32` function remains in
`gemm_32x32_f16_k_loop_helpers.mlir` (section 9) for reference but is not used.

---

### 2. Relax pipeline depth cap for 32x32 ❌ reverted — hurts top configs

Original cap in `_make_substitutions_32x32`:
```python
max_stages_for_2wg = 32768 // cfg.total_lds_bytes
num_stages = min(4, max_stages_for_2wg, max(1, k_outer_iters))
```

For 32x32 with `m_tile=256, n_tile=256, k_t=1` the LDS is 32 KB, giving
`max_stages_for_2wg=1` — no pipelining. The idea was to allow single-WG-per-CU
LDS (64 KB) to unlock depth-2 pipelining for such configs.

**Root cause of regression:** Configs with `16 KB < total_lds_bytes ≤ 32 KB` are
promoted from depth-1 (2 WGs/CU) to depth-2 (1 WG/CU). The top-performing config
(`m_tile=256, n_tile=256, k_t=1`, 32 KB LDS) lives exactly in this range. Going
from 2 WGs/CU to 1 WG/CU reduces the GPU's ability to hide memory-access and
arithmetic latency via wave-level parallelism. For 32×32 tiles, the kernel is not
sufficiently compute-bound to compensate — the occupancy loss dominates and
regresses peak TFLOPS from 470 → 410.

---

### 3. Increase k_tile for 32x32 (arithmetic intensity)

rocBLAS on MI300X uses `k_tile=64` or `k_tile=128` for 32x32. With larger `k_tile`:
- More MFMAs per barrier/LDS-write round → higher ALU utilisation.
- Arithmetic intensity scales as `(m_tile × n_tile × k_tile) / (LDS_bytes × num_barriers)`.

Try adding `k_t=8` (k_tile=256) to `_K_T_VALUES` in the sweep for 32x32. For
`m_tile=128, n_tile=64, k_tile=256`: LDS=32 KB, intensity≈256 flops/byte (matches
MI300X peak compute/bandwidth ratio of ~270 flops/byte for fp16).

**Expected gain:** 5–15% on 32x32.

---

### 4. Verify A-register blocking survives compilation

In `@k_wait_and_compute_mfmas_2d_32x32` the loop order is `(k, m, n)`. For fixed
`(k, m)`, A[k][m] is used for all `n_t_per_wave` n-iterations. The `get_lds_read_value_vx2`
no-op-on-second-call semantics should keep `a_val` in VGPR. Confirm with
`--print-asm` that no spurious `ds_read` or `scratch_load` instructions are emitted
for A inside the n-loop. If spilling occurs, explicitly hold `a_val` in a local
variable outside the n-loop.

---

### 5. Swizzle sweep for 32x32

With `_M_WG_BASE=38`, valid swizzles (divisors of M_WG) are 1, 2, 19, 38. The
current sweep only runs `swizzle=1`. Adding `swizzle=2` to the sweep for 32x32 is
cheap to try (same tiles, different block ordering for L2 B-tile reuse).

---

### 6. Buffer loads (MUBUF) instead of flat loads ✅ applied — +12.5 to +24%

Replaced `global_load_dwordx4` (flat addressing, 2-VGPR address) with
`buffer_load_dwordx4` (MUBUF, 4-SGPR SRD + 1-VGPR offset) for the A and B
global-to-register loads and for the C store.

**Files changed:**
- `_get_library_paths()` in `gemm_fp16_lds.py` switched from
  `global_16x64_b.mlir` + `compute_16x16_f16.mlir` to
  `global_16x64_b_buf.mlir` + `compute_16x16_f16_buf.mlir`.

**Results** (4864×4096×8192, swizzle=1):

| Config                                | Before (TFLOPS) | After (TFLOPS) |
|---------------------------------------|----------------|----------------|
| mt=128, nt=128, kt=64, nw=4, nm=2     | ~424            | 522–528        |
| mt=128, nt=64, kt=64, nw=4, nm=2      | ~400            | ~480           |

**Why it helps:**
- `buffer_load_dwordx4` uses 4 SGPRs (buffer resource descriptor, shared per
  wavefront) + 1 VGPR offset. `global_load_dwordx4` uses 2 VGPRs per address.
  For a wave loading 8 tiles (16 `dwordx4` loads), the MUBUF variant frees
  ~16 VGPRs, relaxing the VGPR pressure and allowing tighter scheduling.
- MUBUF has shorter hardware descriptor paths (no address computation from two
  64-bit registers), reducing scheduler pressure.

**Register budget** (confirmed with `compute_register_budget`):
vgpr + agpr = 131 ≤ 256, LDS = 32 768 bytes → 2 WG/CU fits. The buffer load
savings help stay within the 256 unified-register limit.

---

### 7. CTA swizzle=2 for L2 B-tile reuse ✅ mixed — problem-size dependent

`swizzle=2` reorders CTA blocks so that 2 consecutive M-blocks share the same
N-block. This improves L2 hit rate for the B tile (same B columns loaded by
adjacent CTAs).

**Results** (swizzle=1 → swizzle=2):

| Problem size      | swizzle=1 (TFLOPS) | swizzle=2 (TFLOPS) | Delta  |
|-------------------|-------------------|-------------------|--------|
| 4864×4096×8192    | 498               | 525               | +5.4%  |
| 9728×4096×8192    | ~510              | ~468              | −8%    |

**Why it helps for 4864 but hurts for 9728:**
- For M=4864 with m_wg=38 CTA columns (38×128=4864), swizzle=2 tiles the
  M-axis in pairs → each L2 XCD (4 MB) can cache 2 consecutive B strips.
- For M=9728 (76 columns), the L2 gets thrashed between more CTA groups and
  the B-tile reuse doesn't fit cleanly in the 4 MB XCD L2.

**Recommendation:** for large square GEMMs (M ≈ N ≈ 4864 on MI300X),
`swizzle=2` gives ~5% free. For M >> N or M >> 8192, keep `swizzle=1`.
Valid swizzle values are divisors of `_M_WG_BASE` (38): 1, 2, 19, 38.

---

### 8. Software pipeline depth ≥ 2 ❌ reverted — forces 1 WG/CU

For k_tile=64, LDS = 32 768 bytes per WG. The pipeline-depth cap:
```python
max_stages_for_2wg = 32768 // cfg.total_lds_bytes   # = 1 for k_tile=64
num_stages = min(4, max_stages_for_2wg, ...)
```
Allowing depth-2 would require 65 536 bytes of LDS, so only 1 WG/CU fits.

**Measured** (4864×4096×8192, mt=128, nt=128, kt=64):
- 1 stage (2 WG/CU): **522–528 TFLOPS**
- 2 stages (1 WG/CU): **334 TFLOPS** (−37%)

**Root cause:** the kernel issues 148 instructions per k-tile iteration; the
compute phase is ~256 cycles of MFMAs, far below the ~840-cycle HBM latency
(128 B at 3.35 TB/s). 2 WG/CU is essential to hide this with wave-level
parallelism. At 1 WG/CU the GPU stalls waiting for HBM regardless of the
software pipeline depth.

---

### 9. Compare against rocBLAS / hipBLASLt baseline

Typical MI300X fp16 GEMM performance:
- rocBLAS / hipBLASLt: ~1100–1250 TFLOPS for large square GEMMs.
- Peak fp16: ~1307 TFLOPS.
- Triton AMD: ~950–1050 TFLOPS with `BLOCK=256×256, K=64, stages=2, warps=8`.

Run `hipblas-bench -f gemm -r f16_r --transposeA N --transposeB T -m 4096 -n 4096 -k 4096`
to establish a reference. Then compare with
`python contrib/gemm/bench/bench_gemm_fp16_lds.py --m 4096 --n 4096 --k 4096 ...`.
