// RUN: aster-opt %s --verify-roundtrip

func.func @vop3p_accvgpr_ops(%vdst: !amdgcn.vgpr, %src0: !amdgcn.agpr) {
  %c0 = arith.constant 0 : i32
  %read_vgpr = amdgcn.vop3p v_accvgpr_read_b32 outs %vdst ins %src0 : !amdgcn.vgpr, !amdgcn.agpr
  %write_agpr = amdgcn.vop3p v_accvgpr_write_b32 outs %src0 ins %vdst : !amdgcn.agpr, !amdgcn.vgpr
  amdgcn.vop3p v_accvgpr_write_b32 outs %src0 ins %c0 : !amdgcn.agpr, i32
  return
}
