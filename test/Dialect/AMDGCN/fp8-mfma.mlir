// RUN: aster-opt %s --verify-roundtrip
// Roundtrip test for CDNA3 FP8 MFMA instructions (16x16x32).

!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

func.func @test_fp8_fp8(%dst: !vx4, %a: !vx2, %b: !vx2, %c: !vx4) -> !vx4 {
  %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_fp8> %dst, %a, %b, %c
      : !vx2, !vx2, !vx4 -> !vx4
  return %result : !vx4
}

func.func @test_fp8_bf8(%dst: !vx4, %a: !vx2, %b: !vx2, %c: !vx4) -> !vx4 {
  %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_bf8> %dst, %a, %b, %c
      : !vx2, !vx2, !vx4 -> !vx4
  return %result : !vx4
}

func.func @test_bf8_fp8(%dst: !vx4, %a: !vx2, %b: !vx2, %c: !vx4) -> !vx4 {
  %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf8_fp8> %dst, %a, %b, %c
      : !vx2, !vx2, !vx4 -> !vx4
  return %result : !vx4
}

func.func @test_bf8_bf8(%dst: !vx4, %a: !vx2, %b: !vx2, %c: !vx4) -> !vx4 {
  %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_bf8_bf8> %dst, %a, %b, %c
      : !vx2, !vx2, !vx4 -> !vx4
  return %result : !vx4
}
