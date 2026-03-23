# GEMM fp16 LDS — Investigation Findings & TODOs

**Date:** 2026-03-22
**Kernel:** 16×16 `v_mfma_f32_16x16x16_f16` on gfx942 (MI300X)
**Baseline:** `global_16x64_b_buf.mlir` + `compute_16x16_f16_buf.mlir` (buffer loads, swizzle=2)
**Peak observed:** ~525 TFLOPS on 4864×4096×8192
**MI300X fp16 peak:** ~1307 TFLOPS

---

## Current Kernel Structure (HEAD)

Each K-loop iteration:
1. `alloc_lds` — assigns constant LDS base address for A and B.
2. **Load** k from global into registers (`k_load_a_flat`, `k_load_b_flat`).
3. **Write** registers to LDS (`k_store_a_flat`, `k_store_b_flat`) + wait.
4. **barrier1** — all waves have written LDS[k], safe to read.
5. **Compute** — `k_fused_lds_read_compute_2d` (MFMAs accumulate into AGPRs).
6. `dealloc_lds` + **barrier2** — all waves done reading LDS[k], safe to overwrite.

No prologue. No early loading. Load and write happen sequentially before compute each
iteration. Two barriers per iteration.

---

## Optimizations Applied

### 1. Buffer loads (MUBUF) — **+12.5% to +24%** ✅

Replaced `global_load_dwordx4` (flat address, 2-VGPR address per load) with
`buffer_load_dwordx4` (4-SGPR SRD + 1-VGPR offset) for the A, B global loads and C
store.

**Files:** `_get_library_paths()` in `gemm_fp16_lds.py` now uses
`global_16x64_b_buf.mlir` + `compute_16x16_f16_buf.mlir`.

**Why it helps:** MUBUF frees ~16 VGPRs per wave (8 tile loads × 2 VGPR addresses saved),
relaxing register pressure and allowing tighter scheduling. VGPR + AGPR = 131 ≤ 256 →
still 2 WG/CU.

| Config (4864×4096×8192)              | Before | After     |
|--------------------------------------|--------|-----------|
| mt=128, nt=128, kt=64, nw=4, nm=2   | ~424 TFLOPS | 522–528 TFLOPS |
| mt=128, nt=64, kt=64, nw=4, nm=2    | ~400 TFLOPS | ~480 TFLOPS |

### 2. CTA swizzle=2 — **+5.4% for M=4864** ✅

Reorders CTA blocks so 2 consecutive M-blocks share the same N-block, improving L2
B-tile hit rate.

| Problem size   | swizzle=1 | swizzle=2 | Delta |
|----------------|-----------|-----------|-------|
| 4864×4096×8192 | 498 TFLOPS | 525 TFLOPS | +5.4% |
| 9728×4096×8192 | ~510 TFLOPS | ~468 TFLOPS | −8% |

**Recommendation:** use swizzle=2 only for M ≈ N ≈ 4864. For M >> 8192, keep swizzle=1.
Valid swizzle values are divisors of `_M_WG_BASE` (38): 1, 2, 19, 38.

### 3. Intra-step prefetch for 32×32 — **−13% (reverted)** ❌

Porting the fused LDS-read-compute pattern to 32×32 caused a regression because
`k_inner=4` for 32×32 already saturates the LDS pipeline upfront (16 reads in ~32 cycles
vs. ~20-cycle latency), so the prefetch adds overhead without benefit.

### 4. Relax pipeline depth cap for 32×32 — **−37% (reverted)** ❌

Going from 2 WG/CU to 1 WG/CU dominates. 32×32 tiles are not compute-bound enough to
compensate for the occupancy loss.

### 5. Software pipeline depth ≥ 2 — **−37% (reverted)** ❌

k_tile=64 → 32 KB LDS → only 1 WG/CU at depth 2. 2 WG/CU wave-level parallelism is
essential to hide the ~840-cycle HBM latency. At 1 WG/CU the GPU stalls regardless of
software pipeline depth.

---

## Early-Load Pipeline Investigation (Not Landed)

### Goal

Issue global loads for `k+K_T` right after the barrier (while MFMAs run), overlapping
HBM latency (~100-cycle L2, ~840-cycle HBM miss) with ~256 cycles of MFMA compute.
Expected gain: **22–26%** → targeting **640–660 TFLOPS**.

### Attempted Structure

```
Prologue:
  global_load(k=0) → write LDS[0] → wait → barrier

Loop (k = 0 .. K_TILES, step K_T):
  alloc_lds
  early_global_load(k_next = k+K_T)   ← issued immediately after barrier
  compute(LDS[k])                      ← ~256-cycle MFMA phase hides load latency
  write LDS[k_next]                    ← loads complete by now
  dealloc_lds + barrier
```

One `alloc_lds`/`dealloc_lds` pair per iteration covers both read and write phases.
After `ConvertLDSBuffers`, `get_lds_offset` produces the same constant for both the
prologue and loop allocations (non-overlapping lifetimes → same physical LDS region).

### Bugs Encountered

#### Bug 1: OOB global load on last iteration

The last iteration (`k = K_TILES - K_T`) issues an early load for
`k_next = K_TILES` which is out of bounds. `make_raw_buffer_rsrc` uses
`num_records = 0xFFFFFFFF` — no hardware bounds check — so the load accesses real
memory at a wrong-row address. For large matrices this address falls within the
allocated buffer, causing no page fault but corrupt data written to LDS.

**Symptom:** Illegal memory access / abort for large multi-WG configurations.

**Attempted fix (Option C):** Loop runs `k = 0 .. K_TILES - K_T` (one fewer iteration)
with a separate epilogue block for the last tile. This was abandoned due to the software
pipeliner compatibility issues below.

**Attempted fix (clamp):** Clamp `k_next = min(k + K_T, K_TILES - K_T)` via
`arith.minsi`. Eliminated the OOB access but the last iteration's early load then
re-fetches the current tile (data written to LDS but never read — harmless). This was
also abandoned for the same reasons.

#### Bug 2: Wrong results for K divisible by 256 (= 4 × k_tile for k_tile=64)

With the early-load structure, correctness tests fail for K=256, 512, 768, … (where
`K / k_tile` is divisible by 4). Bad elements appear in rows 0..63 for waves with odd
`wn` (waves 1 and 3). The root cause is the same OOB last-iteration load (Bug 1)
writing wrong-row HBM data into LDS, coincidentally corrupting the memory region that
those specific waves read.

Both bugs are fixed simultaneously by eliminating the OOB last-iteration load — either
via Option C or the clamp approach.

### Software Pipeliner Compatibility Issues (Blocked the Landing)

The early-load structure requires the software pipeliner to work correctly for
performance. Attempts to enable `num_stages ≥ 2` consistently hit compiler errors:

**Problem 1 — Cross-stage use-before-define:**
`get_lds_offset` is annotated `{sched.stage = STAGE_DS_READ}` (stage 2 for `num_stages=4`),
but `k_store_a_flat` uses the result `base_a` and is annotated `STAGE_DS_WRITE` (stage 1).
Stage 1 < stage 2 → the pipeliner rejects the IR: *"cross-stage value defined in stage 2
used in stage 0/1"*.

Moving `alloc_lds`/`get_lds_offset` to `STAGE_GLOBAL_LOAD` (stage 0) eliminates that
error, but then breaks numerical correctness for K=128 (all-zero outputs — stage 0 for
the constant base interferes with the inlined compute's stage annotations in an
unexpected way).

**Problem 2 — MFMA accumulator rotate_head:**
With `num_stages=2`, the MFMA ops carry `sched.rotate_head`. For the epilogue structure
(Option C), the epilogue's MFMA ops are outside the `scf.for` and have no stage
annotation (default stage 0). The pipeliner sees the loop's stage-1 MFMA result used in
stage 0 of the epilogue: *"cross-stage value defined in stage 1 used in stage 0"*.

**Problem 3 — AGPR data layout error for ≥2 loop iterations:**
With `num_stages=1` (all annotations = 0, no pipelining) and the epilogue structure
(Option C), configs with `loop_iters ≥ 2` crash during compilation with:
*"LLVM ERROR: neither the scoping op nor the type class provide data layout information
for `!amdgcn.agpr<[? + 4]>`"*.
This happens because the epilogue's `k_fused_lds_read_compute_2d` (with AGPR outputs)
is outside the `scf.for`. For 1-iteration loops the compiler can scalar-substitute the
AGPR through the loop, but for ≥2 iterations it cannot and falls back to querying the
data layout of the AGPR memref type, which is not registered.

**Root cause summary:** The early-load kernel structure (with or without the epilogue) is
incompatible with the aster software pipeliner as currently implemented. The stage
annotation scheme (`STAGE_GLOBAL_LOAD < STAGE_DS_WRITE < STAGE_DS_READ < STAGE_COMPUTE`)
contradicts the kernel's data flow where `get_lds_offset` (logically a constant,
annotated DS_READ) is consumed by `k_store_a_flat` (annotated DS_WRITE). Without
multi-stage pipelining there is no performance benefit over the baseline (the entire
point of early loading is to overlap the ~100-cycle L2 load with MFMAs via the pipeliner
or at minimum via the `num_stages=2` prologue separation).

### Conclusion

The early-load optimization is not landed. The kernel was fully reverted to the clean
non-pipelined HEAD version. The buffer-load optimization (+12.5–24%) and CTA swizzle
(+5.4%) remain applied.

---

## Register / Resource Budget (Best Config)

Config: 4864×4096×8192, mt=128, nt=128, kt=64, nw=4, nm=2, swizzle=2

```
vgpr = 104, agpr = 64, lds = 32768 bytes, sgpr = 19
Total vgpr+agpr = 168 ≤ 256  →  2 WG/CU fits
```

Assembly (`/tmp/asm_output.txt`):
- Loop counter: `s0` (step=2), condition: `s_cmp_lt_i32 s0, 256`
- LDS write addresses `v82..v97`: computed once before the loop, never clobbered
- LDS read addresses `v100..v103`: computed once after first barrier, never clobbered
- All 64 AGPRs (`a0..a63`) zeroed in prologue

---

## Buffer Descriptor Note

`make_raw_buffer_rsrc` (in `global_16x64_b_buf.mlir`) creates a buffer descriptor with
`num_records = 0xFFFFFFFF` and no range check. OOB `buffer_load` accesses do NOT return
zero — they access real memory at the out-of-bounds address. This is safe for in-bounds
accesses but must be kept in mind when considering any "just load garbage on last iter"
strategy.

---

## TODOs

### P0 — Early-Load Pipeline (Next Attempt)

- [ ] **Understand `sched.stage` semantics** for `alloc_lds`/`get_lds_offset` in the
  pipeliner. The constant-producing ops need a stage ≤ every consumer's stage. Check
  whether annotating them with `STAGE_GLOBAL_LOAD` is semantically valid for the LDS
  allocator (do alloc/dealloc need to bracket the DS_WRITE ops or can they be at any
  stage?).

- [ ] **Fix the cross-stage DS_WRITE < DS_READ contradiction.** Options:
  - Annotate `k_store_a_flat` calls with `{sched.stage = {{STAGE_DS_READ}} : i32}`
    (same stage as `get_lds_offset`), if the pipeliner allows DS_WRITE and DS_READ at
    the same stage.
  - Restructure the stage config table to have `DS_WRITE = DS_READ` for `num_stages ≥ 3`
    (collapse to `num_stages=2` effective stages).
  - Use a single-stage entry point into the kernel and rely on the intra-iteration
    early-load ordering (no software pipeline, but early-load still hides latency).

- [ ] **Resolve the AGPR data layout LLVM error** for `loop_iters ≥ 2` with the
  epilogue structure. The LLVM backend cannot compute `sizeof(agpr<? + 4>)` when the
  AGPR type appears in a memref outside the scf.for. Possible fixes:
  - Use the clamp approach (no epilogue): all ops stay inside the scf.for.
  - Investigate whether `aster-constexpr-expansion` fully scalarizes `%C_buf` for
    ≥2-iteration loops and whether this is the actual failure point.

- [ ] **Benchmark the early-load kernel once it compiles cleanly**, measuring actual
  overlap between HBM latency and MFMA compute. Target: ≥ 640 TFLOPS.

### P1 — 32×32 Kernel Performance

- [ ] **Increase k_tile for 32×32** (`perf_suggestions.md §3`): add `k_t=8` (k_tile=256)
  to the 32×32 sweep. rocBLAS uses k_tile=64–128; larger tiles raise arithmetic intensity
  toward the MI300X peak ratio (~270 FLOP/byte fp16). Expected: +5–15%.

- [ ] **swizzle=2 for 32×32** (`perf_suggestions.md §5`): add `swizzle=2` to the 32×32
  sweep. Same logic as 16×16: divisors of `_M_WG_BASE` (38) = {1, 2, 19, 38}. Cheap
  to try.

- [ ] **Verify A-register blocking** (`perf_suggestions.md §4`): confirm no spurious
  `ds_read` or `scratch_load` for A inside the n-loop in the 32×32 assembly. Use
  `--print-asm` and look for `ds_read` between the first and last MFMA of a k-step.

### P2 — Baseline Comparison

- [ ] **Measure rocBLAS / hipBLASLt** (`perf_suggestions.md §9`):
  ```
  hipblas-bench -f gemm -r f16_r --transposeA N --transposeB T -m 4096 -n 4096 -k 4096
  ```
  Expected: 1100–1250 TFLOPS. Current gap is ~2.5×. Understand which micro-architectural
  bottleneck accounts for most of it (memory bandwidth, occupancy, instruction latency).

- [ ] **Profile with rocprof**: measure `SQ_INSTS_VALU`, `SQ_WAIT_INST_LDS`,
  `TA_TOTAL_WAVEFRONTS`, L2 hit rate for the best 16×16 config to quantify the actual
  bottleneck before the next optimization round.

### P3 — Correctness Coverage

- [ ] **Extend the test suite** to cover:
  - K values where `K / k_tile` is divisible by 4 (256, 512 with k_tile=64) — these
    already pass in HEAD after the pipeline revert, but add them as permanent regression
    guards. *(3 tests added this session: `k256-kt64`, `k512-kt64`, `k256-kt64` with 4
    waves.)*
  - Large multi-WG configs (m=128+, n=128+, multi-block) to catch the class of crash
    that the early-load OOB triggered.
  - 32×32 MFMA variant test cases.

### P4 — Code Quality

- [ ] **Remove unused helper functions** from `gemm_16x32_f16_k_loop_helpers.mlir`:
  - `@k_load_a_flat_into` / `@k_load_b_flat_into` (cross-iteration future approach,
    abandoned because `!future_global_read` cannot cross `scf.for` boundaries).
  - `@k_fused_lds_read_compute_2d_32x32` (section 9, intra-step prefetch for 32×32,
    reverted after −13% regression).

- [ ] **Update stale comment** in `gemm_fp16_lds.mlir`:
  *"LDS is allocated inside the loop so the pipeline can double-buffer (depth 2)"* —
  the pipeline is not currently active; this comment is misleading.

- [ ] **Update `perf_suggestions.md`** with sections for the buffer-load optimization
  (§6) results and the early-load investigation outcome, so the full picture is in one
  place.

---

## Key Numbers at a Glance

| Metric                         | Value                        |
|-------------------------------|------------------------------|
| MI300X peak fp16              | ~1307 TFLOPS                 |
| rocBLAS/hipBLASLt (large sq.) | ~1100–1250 TFLOPS            |
| Triton AMD (BLOCK=256, K=64)  | ~950–1050 TFLOPS             |
| Our best (buf + swizzle=2)    | ~525 TFLOPS (4864×4096×8192) |
| Compute/bandwidth ratio       | ~270 FLOP/byte (fp16, MI300X)|
| L2 latency                    | ~100 cycles                  |
| HBM latency                   | ~840 cycles (128B @ 3.35 TB/s)|
| MFMA compute phase            | ~256 cycles (128nt×128mt tile)|
| LDS latency (`ds_read_b64`)   | ~20 cycles                   |
| 2 WG/CU LDS budget            | 32 768 bytes                 |
