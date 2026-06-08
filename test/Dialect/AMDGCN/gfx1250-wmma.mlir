// RUN: aster-opt %s --verify-roundtrip

// vx8 = 8-VGPR range (v8f32 accumulator, or 16xf16/bf16 packed in 8 VGPRs).
!vx8 = !amdgcn.vgpr<[? + 8]>

func.func @test_wmma_f16(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

func.func @test_wmma_bf16(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_bf16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

func.func @test_wmma_f16_inline_c(%dst: !vx8, %a: !vx8, %b: !vx8) -> !vx8 {
  %c = arith.constant 0.0 : f32
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, f32)
  return %result : !vx8
}

func.func @test_wmma_f16_reuse(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      matrix_a_reuse(unit) matrix_b_reuse(unit)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

func.func @test_wmma_f16_neg(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      neg_lo(array<i1: false, false, true>) neg_hi(array<i1: false, false, true>)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}
