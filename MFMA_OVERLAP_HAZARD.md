# XDL MFMA Overlap Hazard: Root Cause Analysis

## Problem

The following command produces wrong numerical results:

```
python contrib/gemm/gemm_fp16_lds.py \
  --m 4864 --n 4096 --k 4096 \
  --m-tile 128 --n-tile 128 --k-tile 64 \
  --num-waves 8 --num-m-waves 1 \
  --verify --nops 0
```

Relative error: ~9e-2 (catastrophically wrong). The same command with `--nops 5` (or any value ≥ 5) passes verification.

---

## Root Cause

**Missing implementation of `XdlWriteVgprXdlReadSrcCOverlapHazardAttr` in Aster's `amdgcn-hazards` pass.**

### The Hardware Hazard (CDNA3 ISA Section 7.5, Table 37)

On gfx942, consecutive XDL MFMA instructions have two distinct cases for the SrcC register dependency:

| Case | Condition | Required VALU NOPs |
|------|-----------|-------------------|
| **Exact** | Previous MFMA VDST == current MFMA SrcC (same register) | 0 |
| **Overlap** | Previous MFMA VDST ≠ current MFMA SrcC (any other register) | **5** |

For `V_MFMA_F32_16X16X16_F16` on gfx942, the scheduling model assigns `Write4PassMAI` (latency = 4 cycles). The overlap nop count follows:

```
GFX940_XDL_N_PassWritesVGPROverlappedXDLOrSMFMASrcCWaitStates(NumPasses=4, IsGFX950=false)
  = NumPasses + 1 = 5
```

The **exact case** (accumulate-in-place, VDST == SrcC) allows immediate reuse because the hardware forwards the result internally. Any **other accumulator** as SrcC requires 5 VALU NOPs because the XDL write-back bus is still occupied from the previous operation.

### The Kernel's MFMA Pattern

With `--num-m-waves 1`, each wave holds 8 accumulator groups. The loop body issues **32 consecutive MFMAs** cycling through `a[0:3]`, `a[4:7]`, ..., `a[28:31]`:

```asm
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x16_f16 a[0:3],  v[2:3],  v[66:67], a[0:3]   ; writes a[0:3]
v_mfma_f32_16x16x16_f16 a[4:7],  v[6:7],  v[66:67], a[4:7]   ; OVERLAP: needs 5 VALU NOPs
v_mfma_f32_16x16x16_f16 a[8:11], v[10:11], v[66:67], a[8:11]  ; OVERLAP: needs 5 VALU NOPs
... (5 more)
v_mfma_f32_16x16x16_f16 a[28:31], v[30:31], v[66:67], a[28:31]
s_waitcnt lgkmcnt(2)                   ← 1 SALU inst, provides 0 VALU NOPs
v_mfma_f32_16x16x16_f16 a[0:3],  v[4:5],  v[68:69], a[0:3]   ; OVERLAP: needs 5 VALU NOPs
...
```

Every consecutive MFMA pair uses a **different** accumulator → all 31 transitions per loop iteration trigger the overlap hazard. The `s_waitcnt` instructions between k-groups are **scalar** instructions, which do not satisfy the VALU NOP requirement.

### Why the Single-Wave Case Passes

With `--num-waves 1` (single accumulator `a[0:3]`), every MFMA reuses exactly the same accumulator → exact case → 0 NOPs required → no hazard. The kernel passes without `--nops`.

### Why Large Inputs Expose the Bug

The hazard manifests consistently only under high GPU occupancy. With 1 workgroup (`--m 128 --n 128`), the XDL pipeline has slack and the timing violation does not reliably corrupt results. With 1216 workgroups (`--m 4864 --n 4096`), the XDL unit is fully pipelined and the violation is deterministic.

---

## Empirical Confirmation

| `--nops` | Result |
|----------|--------|
| 0 | FAIL (relative error ~1.7M) |
| 2 | FAIL (3.8e-2) |
| 4 | FAIL (4.2e-3) |
| **5** | **PASS** ✓ |
| 6–9 | PASS |

The threshold of **5** matches exactly `GFX940_XDL_N_PassWritesVGPROverlappedXDLOrSMFMASrcCWaitStates(4, false) = 5`.

---

## The Bug in Aster

The hazard is **defined in the `.td` file but not implemented** in `lib/Dialect/AMDGCN/IR/Hazards.cpp`.

### 1. `matchInst` returns false

```cpp
// lib/Dialect/AMDGCN/IR/Hazards.cpp:851
bool CDNA3XdlWriteVgprXdlReadSrcCOverlapHazardAttr::matchInst(
    const InstMetadata *, ISAVersion) const {
  return false; // TODO: XDL/V_SMFMA* write VGPR detection.
}
```

### 2. Hazard raiser not registered

```cpp
// lib/Dialect/AMDGCN/IR/Hazards.cpp:1239 (getHazardRaisersFor, CDNA3)
hazardRaisers.push_back(CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::get(ctx));
hazardRaisers.push_back(CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::get(ctx));
hazardRaisers.push_back(CDNA3XdlWriteVgprVmemValuHazardAttr::get(ctx));
// ↑ XdlWriteVgprXdlReadSrcCOverlapHazardAttr is NOT registered
```

### 3. AGPR outputs skipped in `NonDLOpsValuMfmaHazardAttr`

```cpp
// lib/Dialect/AMDGCN/IR/Hazards.cpp:725
// Only VGPR outputs - MFMA reads VGPR (AGPR uses different path).
if (regTy.getRegisterKind() != RegisterKind::VGPR)
    continue;
```

This comment refers to a "different path" for AGPR-based MFMAs that is currently unimplemented.

### 4. Existing `isHazardTriggered` bug (separate issue)

`XdlWriteVgprXdlReadSrcCExactHazardAttr::isHazardTriggered` checks type equality instead of value identity, causing it to fire for ALL MFMA→MFMA pairs with matching accumulator types (not just exact same registers). This is masked by the hazard having 0-nop counts for case 1, so no incorrect nops are inserted. It is still a correctness concern for multi-pass cases.

---

## What Needs to Be Fixed

### Primary fix: implement `XdlWriteVgprXdlReadSrcCOverlapHazardAttr`

The semantics of "overlap" in the CDNA3 ISA cover **any non-exact accumulator reuse** — the term refers to the temporal overlap of two MFMA operations in the XDL pipeline, not to physical register range overlap. Concretely:

- **Raise** a hazard with the appropriate nop count (case 1 = 5 VALU NOPs for 4-pass MFMA) on the MFMA's VDST register.
- **Trigger** when the next MFMA's SrcC is **not the exact same register** as the hazard-raising MFMA's VDST.
- **Register** the raiser in `getHazardRaisersFor` for CDNA3.

The key distinction from `XdlWriteVgprXdlReadSrcCExactHazardAttr`:

| Hazard | Trigger condition | NOPs (case 1, 4-pass) |
|--------|------------------|----------------------|
| Exact | SrcC == VDST (same value) | 0 |
| Overlap | SrcC != VDST (any other register, including disjoint) | 5 |

Together they cover all MFMA→MFMA SrcC cases.

### Secondary: add `XdlWriteVgprXdlReadSrcCOverlapHazardAttr` to unit tests

The existing tests in `test/Dialect/AMDGCN/Transforms/hazards-cdna3-mma.mlir` only use VGPR accumulators (`!amdgcn.vgpr<[? + 4]>`). Tests using AGPR accumulators (`!amdgcn.agpr<[? + 4]>`) with different accumulator groups need to be added to cover the pattern in this kernel.

---

## Relevant Files

| File | Relevance |
|------|-----------|
| `lib/Dialect/AMDGCN/IR/Hazards.cpp` | Hazard implementations; `matchInst`, `populateHazardsFor`, `isHazardTriggered`, `getHazardRaisersFor` |
| `include/aster/Dialect/AMDGCN/IR/Hazards.td` | Hazard definitions and nop counts per case |
| `lib/Dialect/AMDGCN/Transforms/AMDGCNHazards.cpp` | NOP insertion pass; reads `flushedInstCounts` |
| `lib/Dialect/AMDGCN/Analysis/HazardAnalysis.cpp` | Forward dataflow analysis |
| `test/Dialect/AMDGCN/Transforms/hazards-cdna3-mma.mlir` | Unit tests (all VGPR-based, missing AGPR overlap tests) |
| `contrib/kittens/library/compute_16x16_f16.mlir` | MFMA primitive using AGPR accumulators |
