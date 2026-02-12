// RUN: aster-opt %s --verify-roundtrip

//===----------------------------------------------------------------------===//
// VOP1 Conversion Operations
//===----------------------------------------------------------------------===//

func.func @test_vop1_cvt_f32_f16_vgpr(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_f16> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_f32_f16_sgpr(%src0: !amdgcn.sgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_f16> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_f16_f32(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f16_f32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_f32_u32(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_u32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_f32_i32(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_i32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_u32_f32(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_u32_f32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_cvt_i32_f32(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_i32_f32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

// Test with immediate source
func.func @test_vop1_cvt_f32_u32_imm(%dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c42 = arith.constant 42 : i32
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_cvt_f32_u32> %dst, %c42
      : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// VOP3 Pack Operations
//===----------------------------------------------------------------------===//

func.func @test_vop3_pack_b32_f16(%vdst: !amdgcn.vgpr, %src0: !amdgcn.vgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop3 v_pack_b32_f16 outs %vdst ins %src0, %src1
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop3_pack_b32_f16_sgpr_src(%vdst: !amdgcn.vgpr, %src0: !amdgcn.sgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop3 v_pack_b32_f16 outs %vdst ins %src0, %src1
      : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop3_pack_b32_f16_imm_src(%vdst: !amdgcn.vgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %result = amdgcn.vop3 v_pack_b32_f16 outs %vdst ins %c0, %src1
      : !amdgcn.vgpr, i32, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// VOP3 FP8/BF8 Pack-Convert Operations
//===----------------------------------------------------------------------===//

func.func @test_vop3_cvt_pk_fp8_f32(%vdst: !amdgcn.vgpr, %src0: !amdgcn.vgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop3 v_cvt_pk_fp8_f32 outs %vdst ins %src0, %src1
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop3_cvt_pk_bf8_f32(%vdst: !amdgcn.vgpr, %src0: !amdgcn.vgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop3 v_cvt_pk_bf8_f32 outs %vdst ins %src0, %src1
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop3_cvt_pk_fp8_f32_sgpr(%vdst: !amdgcn.vgpr, %src0: !amdgcn.sgpr, %src1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop3 v_cvt_pk_fp8_f32 outs %vdst ins %src0, %src1
      : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}
