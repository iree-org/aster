# GFX942 GEMM FP16 LDS — Performance Analysis and Suggestions

**Config evaluated:** `--m 4864 --n 4096 --k 8192 --m-tile 128 --n-tile 128 --k-tile 64 --num-waves 4 --num-m-waves 4`

**Result:** 407.9 TFLOP/s
**Theoretical peak** (228 CUs @ ~1.9 GHz, 1 MFMA per 8 cycles per CU): ~443 TFLOP/s
**Efficiency:** ~92%

---

## Suggestions by Order of Importance

### 1. Enable software pipelining ✓ *implemented (LDS-aware)*

The 32×32 path already selects a depth-4 pipeline based on `k_outer_iters`. The 16×16
path had all stages hardcoded to `0`.

A naïve port (just mirroring the 32×32 logic) fails for the top config: with k_tile=64,
a 2-stage pipeline doubles LDS from 32 KB to 64 KB — filling the entire 64 KB per-CU
budget and reducing occupancy from 2 workgroups/CU to 1. Net result is a **regression**
(305 TFLOP/s vs 419 TFLOP/s).

The fix caps pipeline depth so that `num_stages × total_lds_bytes ≤ 32768`, preserving
2-workgroup-per-CU occupancy:

```python
max_stages_for_2wg = 32768 // cfg.total_lds_bytes
num_stages = min(4, max_stages_for_2wg, max(1, k_outer_iters))
```

This selects 1-stage (no pipelining) for k_tile=64 — the right answer — and 2-stage
for k_tile=32, where pipelining genuinely helps.

**Why pipelining helps k_tile=32 but not k_tile=64:**

| Config | MFMAs/step | MFMA cycles | HBM latency | Gap hidden by 2-WG? | Pipeline benefit |
|--------|-----------|-------------|-------------|----------------------|-----------------|
| k_tile=64 | 64 | ~512 cy | ~300 cy | Yes (512 > 300) | None — overhead only |
| k_tile=32 | 32 | ~256 cy | ~300 cy | No (256 < 300) | Yes — closes the gap |

With k_tile=64, the 512-cycle compute window already exceeds HBM latency, so 2-WG
overlap fully hides memory access and pipelining only adds prologue/epilogue overhead.
With k_tile=32, the 256-cycle window is shorter than HBM latency; pipelining prefetches
the next iteration's tiles during the current compute, closing the stall.

**Measured results (2×2 wave grid throughout):**

| Config | LDS/WG | VGPRs | Regs total | Waves/SIMD | WGs/CU | TFLOP/s |
|--------|--------|-------|------------|------------|--------|---------|
| k_tile=64, no pipeline | 32 KB | 97 | 164 | 3 | 2 | 416.5 |
| k_tile=32, no pipeline | 16 KB | — | — | 3 | 2 | 344.8 |
| k_tile=32, 2-stage pipeline | 32 KB | 84 | 148 | 3 | 2 | **362.2** |

Pipelining k_tile=32 gives **+5.1%** (344.8 → 362.2 TFLOP/s) while preserving both
3-waves/SIMD and 2-WGs/CU (148 total regs, 32 KB LDS). It does not close the gap with
k_tile=64 (416.5 TFLOP/s) because k_tile=32 runs 2× as many outer iterations, each
with proportionally more barrier and loop-counter overhead relative to MFMA work.

---

### 2. Intra-step LDS read prefetch ✓ *implemented*

All `m_t_per_wave × k_inner` A reads and `n_t_per_wave × k_inner` B reads were
previously issued by `k_read_lds_a_2d` / `k_read_lds_b_2d` before any MFMA,
producing a `s_waitcnt lgkmcnt(N)` countdown stall at the start of
`k_wait_and_compute_mfmas_2d`. With k_tile=64 (k_inner=8, m_t=n_t=4), that is
64 reads in-flight, each ~20–40 cycles — a staircase of stalls across the first
~60 MFMAs.

The fix fuses the three calls into `k_fused_lds_read_compute_2d`, which issues
reads for ki+1 just before resolving ki:

```
Prologue: issue reads for ki=0
for ki in [0, k_inner):
    if ki+1 < k_inner: issue reads for ki+1   ← prefetch
    resolve ki reads + compute m_t × n_t MFMAs
    advance cur ← next
```

At each `s_waitcnt` only `m_t_per_wave + n_t_per_wave = 8` reads are newly in-flight
(ki+1), vs. 64 before, so the stall is near-zero for all but ki=0.

**Measured results (k_tile=64, 2×2 wave grid, 3 warmup + 15 iterations, 3 runs):**

| Config | TFLOP/s |
|--------|---------|
| Before (k_read_lds + k_wait_and_compute) | ~418.8 |
| After (k_fused_lds_read_compute_2d)      | ~422–430 |

**Observed gain: +1–3%** (vs. predicted 6–8%). The gap between predicted and actual
is likely because `sched.rotate_head` / `sched.stage` annotations already enable
some intra-stage instruction reordering, partially hiding the stall in the original
code. The aster software pipeliner may also be issuing some reads earlier than the
MLIR source order suggests.

---

### 3. Switch `ds_write_b64 × 2` → `ds_write_b128 × 1`

`store_global_tile_to_lds_16x64_b` writes 16 bytes per thread via two `ds_write_b64`
instructions. A single `ds_write_b128` covers the same 16 bytes. This halves the DS
write instruction count per outer step (16 → 8), halves the lgkmcnt tokens consumed by
writes, and reduces barrier overhead. `ds_write_b128` is available on GFX942.

**Fix:** Replace the two `ds_write_b64` calls in `lds_16x64_b.mlir` with a single
`ds_write_b128`.

---

### 3. Enable CTA swizzle for B-tile L2 reuse

With `swizzle=1` (row-major dispatch), each of the 32 B-tiles (N_WG=32) is consumed by
38 M-blocks in a round-robin pattern, preventing L2 reuse. The B matrix (64 MB) far
exceeds L2 capacity (~8 MB), so each B tile is effectively re-fetched from HBM 38 times.

A swizzle factor of 4–8 groups consecutive flat block IDs onto the same N-tile, keeping
it in L2 across 4–8 M-block computations before moving on.

**Fix:** Pass `--swizzle 4` (or 8) and ensure `m_wg % swizzle == 0` (m_wg=38, so
swizzle must divide 38: try 1, 2, 19, 38; or resize M to make m_wg a power of 2).

---

### 4. Double k_tile to 128 to fill available LDS

Current LDS usage is 32 KB, exactly half the 64 KB maximum. Doubling `k_tile` to 128
sets `k_t=4`, filling all 64 KB. Benefits:
- Halves the outer K-loop iteration count (128 → 64), reducing barrier overhead by ~2×.
- Each outer step processes 8 MFMA K-steps (`k_inner=8`) with 128 MFMAs — a wider
  compute window that better amortises load/store overhead.
- With pipelining enabled, the prefetch window is deeper.

**Fix:** Use `--k-tile 128`.

---

### 5. Add benchmark warmup

The reported average includes iteration 0 at ~960 µs (~20% slower than steady-state
~780 µs due to cold L2/HBM). Use `--num-warmup 2` to exclude cold-start effects and
report the stable throughput.

**Fix:** Pass `--num-warmup 2 --num-iterations 12` (or similar).

---

### 6. Rebalance wave grid from 4×1 to 2×2 ✓ *implemented*

The current `num_m_waves=4` / `num_n_waves=1` grid gives each wave a 2×8 output tile
(`m_t_per_wave=2`, `n_t_per_wave=8`). This is strongly asymmetric: each wave loads 2
A-tiles but 8 B-tiles per K-step, causing uneven register pressure between A and B
registers.

A 2×2 grid (`num_m_waves=2`) gives `m_t_per_wave=4`, `n_t_per_wave=4` — a balanced
4×4 tile per wave with symmetric A/B register usage. The same 16 output tiles per wave
(`tiles_per_wave=16`) are preserved, but the VGPR pressure from the two asymmetric
LDS-read address sequences is equalized. If total registers drop below 170, the hardware
can schedule 3 waves per SIMD instead of 2, gaining an additional latency-hiding slot.

**Fix:** Replace `--num-m-waves 4` with `--num-m-waves 2`.

**Result:**

| Config        | VGPRs | AGPRs | Total regs | Waves/SIMD | TFLOP/s |
|---------------|-------|-------|------------|------------|---------|
| 4×1 (before)  | 112   | 64    | 176        | 2          | 407.9   |
| 2×2 (after)   | 97    | 64    | 164        | 3          | 418.8   |

The 2×2 grid reduced VGPRs from 112 to 97 (register file total: 176 → 164), crossing
the threshold from 2 to 3 waves per SIMD (512 / 164 = 3.12). The 50% occupancy
increase translates to a **+2.7% throughput gain** (407.9 → 418.8 TFLOP/s). The gain
is modest because at ~92% of peak, latency hiding is already effective with 2 workgroups
per CU; further gains require pipelining (suggestion 1).

---

### 7. Evaluate the 32×32×8 MFMA variant

`run_gemm_32x32` (same script, `--mfma-variant 32x32`) already includes software
pipelining and processes 16 384 FLOPs/MFMA vs 8 192 for the 16×16 variant. On GFX942,
the 32×32 variant typically reaches higher efficiency because the FLOPs-per-instruction
ratio improves and barrier/address overhead is better amortised.

**Fix:** Run:
```
python contrib/gemm/gemm_fp16_lds.py \
  --m 4864 --n 4096 --k 8192 \
  --m-tile 128 --n-tile 128 --k-tile 64 \
  --num-waves 4 --num-m-waves 2 \
  --mfma-variant 32x32 \
  --num-warmup 2 --num-iterations 12
```
