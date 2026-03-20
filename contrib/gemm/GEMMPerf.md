# GEMM Performance Analysis

Comparison of `gemm_fp16_lds` against top-performing GEMM kernels (CUTLASS 3.x,
rocBLAS, Triton) on gfx942 (CDNA3 / MI300X), fp16→fp32,
`v_mfma_f32_16x16x16_f16`.

---

## 1. No software pipelining — biggest gap

**Current:** each K outer-loop iteration is entirely sequential:

```
global_load → wait → write_LDS → barrier → read_LDS → compute
```

The stage annotations (`STAGE_GLOBAL_LOAD`, `STAGE_DS_WRITE`, `STAGE_DS_READ`,
`STAGE_COMPUTE` — all set to 0) show this was anticipated but not yet activated.

**What top kernels do:** CUTLASS 3.x, Triton's pipelining, and rocBLAS all
overlap iteration N+1's global loads with iteration N's MFMA compute, using a
**double-buffered LDS** (two ping-pong LDS regions). This hides the full global
memory latency (~300–600 cycles on MI300X) behind the compute phase. For
K-bound problems this is the single largest performance lever — typically
1.5–2× throughput improvement.

**Concretely:** the LDS needs to double in size
(`2 × (A_LDS_BYTES + B_LDS_BYTES)`), and the barrier shifts: issue loads for
step k+1, compute MFMAs for step k, then barrier and swap buffers.

---

## 2. Only 16×16×16 MFMA — suboptimal throughput density

**Current:** exclusively uses `v_mfma_f32_16x16x16_f16` (4096 ops/instruction,
4 AGPRs per accumulator tile).

**What top kernels do:** gfx942 also exposes `v_mfma_f32_32x32x8_f16` (8192
ops/instruction) and `v_mfma_f32_16x16x32_f16` (8192 ops/instruction via b16
packing). The 32×32×8 variant:

- Doubles FMA ops per instruction, reducing instruction-fetch pressure.
- Each accumulator tile grows to 16 AGPRs but covers a 32×32 output region —
  AGPR:compute ratio is better.
- Reduces the number of MFMAs needed per K-tile, easing scheduling.

This is especially relevant when occupancy is limited by VGPR count.

---

## 3. Minimal K-tile granularity — high overhead per unit of useful work

**Current:** the fundamental tile is `16×32` f16 (one `dwordx4` per lane),
meaning K_T=1 implies 32 K-elements per outer step. Even with K_T=2
(K_TILE=64), each outer iteration only issues 2 MFMA K-steps per M-N tile pair.

**What top kernels do:** rocBLAS and CUTLASS default to K-tiles of 64–128
elements (2–4× larger), amortizing the LDS allocation and barrier overhead
across more useful MFMA work per iteration. Larger K-tiles also make pipelining
worth the LDS cost.

Suggestion: normalize toward K_TILE ≥ 64 (K_T ≥ 2) as the minimum tunable
config, and expose K_TILE=128 (K_T=4) as the primary tuning target.

---

## 4. Wave output tile shape is skewed toward M — limits register file efficiency

**Current constraint:**

```
m_tile % (16 * num_waves) == 0
n_tile % (16 * num_waves) == 0
```

Each wave computes `M_T_PER_WAVE × N_T` tiles, where N_T is the *full* N tile
per WG. For a 2-wave config with M_T=2, N_T=4, each wave does 1×4 = 4
accumulator tiles = 16 AGPRs. With M_T=4, N_T=4 and 2 waves, each wave does
2×4 = 8 tiles = 32 AGPRs, leaving ~32 VGPRs for loads — tight.

**What top kernels do:** partition the output tile more squarely per wavefront
using a 2D wave grid (num_m_waves × num_n_waves = num_waves). Each wave (wm,
wn) owns the `M_T_PW × N_T_PW` sub-tile of C. Waves cooperatively load all A
and B tiles into LDS (loading assignment is independent of the 2D compute
partition), then after a barrier each wave reads only its own A-row-band and
B-column-band. This halves AGPR usage per wave in the 2×2 case.

**Implementation note:** requires decomposing the flat wave ID into a 2D
coordinate (wm, wn) and separating the loading assignment from the compute
assignment. Simply restricting B reads after the existing 1D barrier is
incorrect — off-diagonal C tiles would be left uncomputed.

---

## 5. LDS reads and MFMA are not overlapped

**Current:** `@k_read_lds_a` and `@k_read_lds_b` issue all `ds_read_b64`
instructions as a batch, collect futures, then `@k_wait_and_compute_mfmas`
waits each future just before consuming it.

The `future_lds_read` abstraction already carries a read token — this is the
right structure. However, all LDS reads are issued *after* the barrier but
*before* any MFMA. On gfx942, `ds_read_b64` has ~40-cycle latency. Ideally,
reads for MFMA tile `(m, k, n)` should be issued as early as possible so the
wait immediately before that MFMA is already satisfied.

**Suggestion:** restructure `@k_wait_and_compute_mfmas` to issue the LDS read
for `(m, k+1, n)` while waiting for `(m, k, n)` — software pipelining within
the K-MFMA loop itself.

---

## 6. MFMA loop ordering `(m, k, n)` is suboptimal for latency hiding

**Current:** the compute loop linearizes as
`(m_tile_idx, k_mfma_idx, n_tile_idx)`, i.e., outer=m, inner=n with k in the
middle.

**What top kernels do:** the preferred ordering on CDNA is `(k, m, n)` — keep
all N-tiles for a fixed M tile updating together so the MFMA 2–8 cycle latency
on a given accumulator is hidden by the intervening MFMAs on other
accumulators. With N_T=4 tiles and K_T×2=4 MFMA steps, the innermost loop
should be over `n` to maximize the number of different AGPR ranges used
consecutively.

Concretely, changing from `(m, k, n)` to `(k, m, n)` so the K dimension is
outermost within a K-tile step lets the n-loop reuse the same A fragment across
all N tiles before loading the next K-step — matching the CUTLASS "K-major"
MFMA order that reduces register file reads.

---

## 7. No persistent kernel — dispatch overhead for large grids

**Current:** one workgroup = one output tile. For M=4096, N=4096, M_TILE=64,
N_TILE=64: 4096 workgroups launched. On MI300X with 304 CUs, ~13 rounds of
workgroups.

**What top kernels do:** persistent kernels (Triton's `tl.cdiv`-based loop,
rocBLAS internal) keep workgroups alive across multiple output tiles, amortizing
kernel launch overhead and enabling better CU utilization by avoiding the
tail-effect in the final wave of WG launches. This also allows smarter tile
scheduling (e.g., StreamK / even-K splitting) to eliminate the output-tile size
alignment constraints.

---

## 8. C epilogue: non-temporal stores not used

**Current:** `@store_global_C_mfma_f32_16x16x16_f16` stores AGPR-resident f32
accumulators to global C via `flat_store_dwordx4`. For large N, consecutive WGs
writing different M tiles to the same N-column range will thrash L2.

**Suggestion:** use `buffer_store_dwordx4` with the `nt` (non-temporal) hint
for the C epilogue when C is write-only. This is what rocBLAS does for large
streaming GEMMs and avoids polluting the 32MB L2 on MI300X with output data
that will not be reused.

---

## Summary

| Gap | Impact | Complexity |
|-----|--------|------------|
| Software pipelining (double-buffer LDS) | Very high | High — stage annotations already wired |
| `v_mfma_f32_32x32x8_f16` or `16x16x32` | High | Medium — new tile library needed |
| Larger K-tile (K_T ≥ 4 as default target) | Medium-high | Low — config change |
| MFMA loop ordering `(k, m, n)` | Medium | Low — reorder in `k_wait_and_compute_mfmas` |
| LDS read/MFMA overlap within K-MFMA loop | Medium | Medium |
| Per-wave output tile shape (square partition) | Medium | Medium |
| Non-temporal C stores | Low-medium | Low |
| Persistent kernel / StreamK dispatch | Medium | High |

The stage annotations in the template already suggest the pipelining path is
understood. Activating double buffering and switching the MFMA loop ordering to
`(k, m, n)` would be the highest-ROI changes relative to implementation effort.
