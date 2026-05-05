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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1) cbsz(2 : i8) blgp(4 : i8)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1) op_sel_hi(array<i1: true, true, false>)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1)
    : outs(!amdgcn.agpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.agpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_32x32x64_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1)
    : outs(!amdgcn.vgpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 16]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_32x32x64_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1) cbsz(1 : i8) blgp(3 : i8)
    : outs(!amdgcn.agpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.agpr<[? + 16]>, !amdgcn.vgpr, !amdgcn.vgpr)
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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.sgpr)
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
  %result = amdgcn.v_mfma_scale_f32_16x16x128_f8f6f4 outs(%dst) ins(%a, %b, %c, %s0, %s1) cbsz(3 : i8) blgp(4 : i8) op_sel_hi(array<i1: true, true, false>)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 8]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, !amdgcn.vgpr)
  return
}

//===----------------------------------------------------------------------===//
// CDNA4 32x32x16 MFMA (f16 inputs, f32 accumulator)
//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x16_f16 is CDNA4-only. 8-pass instruction with 4 input
// VGPRs per operand and 16 accumulator VGPRs.

func.func @test_mfma_32x32x16_f16_basic(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.vgpr<[? + 16]>, %dst: !amdgcn.vgpr<[? + 16]>) {
  %result = amdgcn.v_mfma_f32_32x32x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 16]>)
  return
}

func.func @test_mfma_32x32x16_f16_agpr(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.agpr<[? + 16]>, %dst: !amdgcn.agpr<[? + 16]>) {
  %result = amdgcn.v_mfma_f32_32x32x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.agpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.agpr<[? + 16]>)
  return
}

//===----------------------------------------------------------------------===//
// CDNA4 16x16x32 MFMA (f16 inputs, f32 accumulator)
//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x32_f16 is CDNA4-only. 4-pass instruction with 4 input
// VGPRs per operand and 4 accumulator VGPRs. Doubled-K variant of 16x16x16.

// CHECK-LABEL: func @test_mfma_16x16x32_f16_basic
// CHECK: amdgcn.v_mfma_f32_16x16x32_f16
func.func @test_mfma_16x16x32_f16_basic(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  %result = amdgcn.v_mfma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>)
  return
}

// CHECK-LABEL: func @test_mfma_16x16x32_f16_agpr
// CHECK: amdgcn.v_mfma_f32_16x16x32_f16
func.func @test_mfma_16x16x32_f16_agpr(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.agpr<[? + 4]>, %dst: !amdgcn.agpr<[? + 4]>) {
  %result = amdgcn.v_mfma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.agpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.agpr<[? + 4]>)
  return
}

//===----------------------------------------------------------------------===//
// CDNA4 16x16x32 MFMA (bf16 inputs, f32 accumulator)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_mfma_16x16x32_bf16_basic
// CHECK: amdgcn.v_mfma_f32_16x16x32_bf16
func.func @test_mfma_16x16x32_bf16_basic(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>) {
  %result = amdgcn.v_mfma_f32_16x16x32_bf16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 4]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>)
  return
}

//===----------------------------------------------------------------------===//
// CDNA4 32x32x16 MFMA (bf16 inputs, f32 accumulator)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_mfma_32x32x16_bf16_basic
// CHECK: amdgcn.v_mfma_f32_32x32x16_bf16
func.func @test_mfma_32x32x16_bf16_basic(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.vgpr<[? + 16]>, %dst: !amdgcn.vgpr<[? + 16]>) {
  %result = amdgcn.v_mfma_f32_32x32x16_bf16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 16]>)
  return
}

// CHECK-LABEL: func @test_mfma_32x32x16_bf16_agpr
// CHECK: amdgcn.v_mfma_f32_32x32x16_bf16
func.func @test_mfma_32x32x16_bf16_agpr(
    %a: !amdgcn.vgpr<[? + 4]>, %b: !amdgcn.vgpr<[? + 4]>,
    %c: !amdgcn.agpr<[? + 16]>, %dst: !amdgcn.agpr<[? + 16]>) {
  %result = amdgcn.v_mfma_f32_32x32x16_bf16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.agpr<[? + 16]>)
      ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.agpr<[? + 16]>)
  return
}
