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

### 6. Compare against rocBLAS / hipBLASLt baseline

Typical MI300X fp16 GEMM performance:
- rocBLAS / hipBLASLt: ~1100–1250 TFLOPS for large square GEMMs.
- Peak fp16: ~1307 TFLOPS.
- Triton AMD: ~950–1050 TFLOPS with `BLOCK=256×256, K=64, stages=2, warps=8`.

Run `hipblas-bench -f gemm -r f16_r --transposeA N --transposeB T -m 4096 -n 4096 -k 4096`
to establish a reference. Then compare with
`python contrib/gemm/bench/bench_gemm_fp16_lds.py --m 4096 --n 4096 --k 4096 ...`.
