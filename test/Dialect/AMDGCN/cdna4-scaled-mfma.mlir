// RUN: aster-opt %s --verify-roundtrip

//===----------------------------------------------------------------------===//
// CDNA4 Scaled MFMA 16x16x128 Operations
//===----------------------------------------------------------------------===//
// V_MFMA_SCALE_F32_16X16X128_F8F6F4 is a single 4-DWORD instruction
// combining ld_scale + MFMA into one encoding.

// Basic scaled 16x16x128 with VGPR scale sources
func.func @test_scaled_mfma_16x16x128_basic(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 4]>,
    %dst: !amdgcn.vgpr<[? + 4]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.vgpr<[? + 4]>
  return
}

// 16x16x128 with format codes: cbsz=2 (fp6 for A), blgp=4 (fp4 for B)
func.func @test_scaled_mfma_16x16x128_formats(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 4]>,
    %dst: !amdgcn.vgpr<[? + 4]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1 cbsz = 2 blgp = 4
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.vgpr<[? + 4]>
  return
}

// 16x16x128 with op_sel on scale sources (byte 3 = bits [31:24])
func.func @test_scaled_mfma_16x16x128_op_sel(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 4]>,
    %dst: !amdgcn.vgpr<[? + 4]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1 op_sel_0 = 3 op_sel_1 = 3
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.vgpr<[? + 4]>
  return
}

// 16x16x128 with AGPR accumulators
func.func @test_scaled_mfma_16x16x128_agpr(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.agpr<[? + 4]>,
    %dst: !amdgcn.agpr<[? + 4]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.agpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.agpr<[? + 4]>
  return
}

//===----------------------------------------------------------------------===//
// CDNA4 Scaled MFMA 32x32x64 Operations
//===----------------------------------------------------------------------===//

// Basic scaled 32x32x64
func.func @test_scaled_mfma_32x32x64_basic(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 16]>,
    %dst: !amdgcn.vgpr<[? + 16]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_32x32x64_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 16]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.vgpr<[? + 16]>
  return
}

// 32x32x64 with AGPR accumulators + format codes
func.func @test_scaled_mfma_32x32x64_agpr_formats(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.agpr<[? + 16]>,
    %dst: !amdgcn.agpr<[? + 16]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_32x32x64_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1 cbsz = 1 blgp = 3
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.agpr<[? + 16]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.agpr<[? + 16]>
  return
}

// Scaled MFMA with SGPR scale sources
func.func @test_scaled_mfma_sgpr_scales(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 4]>,
    %dst: !amdgcn.vgpr<[? + 4]>,
    %s0: !amdgcn.sgpr,
    %s1: !amdgcn.sgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.sgpr
      -> !amdgcn.vgpr<[? + 4]>
  return
}

// All modifiers combined
func.func @test_scaled_mfma_all_modifiers(
    %a: !amdgcn.vgpr<[? + 8]>,
    %b: !amdgcn.vgpr<[? + 8]>,
    %c: !amdgcn.vgpr<[? + 4]>,
    %dst: !amdgcn.vgpr<[? + 4]>,
    %s0: !amdgcn.vgpr,
    %s1: !amdgcn.vgpr) {
  %result = amdgcn.vop3p.vop3p_scaled_mai #amdgcn.inst<v_mfma_scale_f32_16x16x128_f8f6f4>
      %dst, %a, %b, %c, %s0, %s1 op_sel_0 = 1 op_sel_1 = 2 cbsz = 3 blgp = 4
      : !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>,
        !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr
      -> !amdgcn.vgpr<[? + 4]>
  return
}
