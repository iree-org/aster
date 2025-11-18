// RUN: aster-opt %s --verify-roundtrip

func.func @vop_add_ops(%vdst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr,
                       %carry: !amdgcn.sgpr_range<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr,
                                                 !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
  // v_add_co_u32 - VOP2 add with carry out
  %add_co, %carry_out = amdgcn.vop2 v_add_co_u32 outs %vdst dst1 = %carry ins %lhs, %rhs
    : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr

  // v_add_u16 - VOP2 add on 16-bit unsigned operands
  %add_u16 = amdgcn.vop2 v_add_u16 outs %vdst ins %lhs, %rhs
    : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

  // v_add_u32 - VOP2 add on 32-bit unsigned operands
  %add_u32 = amdgcn.vop2 v_add_u32 outs %vdst ins %lhs, %rhs
    : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

  // v_add_i16 - VOP2 add on 16-bit signed operands
  %add_i16 = amdgcn.vop2 v_add_i16 outs %vdst ins %lhs, %rhs
    : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

  // v_add_i32 - VOP2 add on 32-bit signed operands
  %add_i32 = amdgcn.vop2 v_add_i32 outs %vdst ins %lhs, %rhs
    : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

  // v_addc_co_u32 - VOP add with carry in and out
  %addc_co, %carry_out2 = amdgcn.vop2 v_addc_co_u32 outs %vdst dst1 =  %carry ins %lhs, %rhs src2 = %carry
    : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>

  return %add_co, %carry_out, %add_u16, %add_u32, %add_i16, %add_i32
    : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
}
