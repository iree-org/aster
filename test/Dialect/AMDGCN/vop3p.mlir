// RUN: aster-opt %s --verify-roundtrip

func.func @vop3p_accvgpr_ops(%vdst: !amdgcn.vgpr, %src0: !amdgcn.agpr) {
  %c0 = arith.constant 0 : i32
  %read_vgpr = amdgcn.v_accvgpr_read outs(%vdst) ins(%src0)
    : outs(!amdgcn.vgpr) ins(!amdgcn.agpr)
  %write_agpr = amdgcn.v_accvgpr_write outs(%src0) ins(%vdst)
    : outs(!amdgcn.agpr) ins(!amdgcn.vgpr)
  amdgcn.v_accvgpr_write outs(%src0) ins(%c0)
    : outs(!amdgcn.agpr) ins(i32)
  return
}
