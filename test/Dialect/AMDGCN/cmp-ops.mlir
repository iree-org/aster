// RUN: aster-opt %s --verify-roundtrip

func.func @cmpi(%scc: !amdgcn.scc, %vcc: !amdgcn.vcc, %a: i32, %b: i32,
    %v1: !amdgcn.vgpr, %dst: !amdgcn.sgpr, %dstAlloc: !amdgcn.sgpr<0>) {
  amdgcn.cmpi s_cmp_eq_i32 outs %scc ins %a, %b : outs(!amdgcn.scc) ins(i32, i32)
  amdgcn.cmpi v_cmp_eq_i32 outs %vcc ins %a, %v1 : outs(!amdgcn.vcc) ins(i32, !amdgcn.vgpr)
  %0 = amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %a, %v1 : dps(!amdgcn.sgpr) ins(i32, !amdgcn.vgpr)
  amdgcn.cmpi v_cmp_eq_i32_e64 outs %dstAlloc ins %a, %v1 : outs(!amdgcn.sgpr<0>) ins(i32, !amdgcn.vgpr)
  return
}

func.func @sopc_signed_comparisons(%src0: !amdgcn.sgpr, %src1: !amdgcn.sgpr, %scc: !amdgcn.scc) {
  // s_cmp_eq_i32 - SOPC compare equal (signed 32-bit)
  amdgcn.cmpi s_cmp_eq_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_lg_i32 - SOPC compare not equal (signed 32-bit)
  amdgcn.cmpi s_cmp_lg_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_gt_i32 - SOPC compare greater than (signed 32-bit)
  amdgcn.cmpi s_cmp_gt_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_ge_i32 - SOPC compare greater than or equal (signed 32-bit)
  amdgcn.cmpi s_cmp_ge_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_lt_i32 - SOPC compare less than (signed 32-bit)
  amdgcn.cmpi s_cmp_lt_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_le_i32 - SOPC compare less than or equal (signed 32-bit)
  amdgcn.cmpi s_cmp_le_i32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  return
}

func.func @sopc_unsigned_comparisons(%src0: !amdgcn.sgpr, %src1: !amdgcn.sgpr, %scc: !amdgcn.scc) {
  // s_cmp_eq_u32 - SOPC compare equal (unsigned 32-bit)
  amdgcn.cmpi s_cmp_eq_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_lg_u32 - SOPC compare not equal (unsigned 32-bit)
  amdgcn.cmpi s_cmp_lg_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_gt_u32 - SOPC compare greater than (unsigned 32-bit)
  amdgcn.cmpi s_cmp_gt_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_ge_u32 - SOPC compare greater than or equal (unsigned 32-bit)
  amdgcn.cmpi s_cmp_ge_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_lt_u32 - SOPC compare less than (unsigned 32-bit)
  amdgcn.cmpi s_cmp_lt_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  // s_cmp_le_u32 - SOPC compare less than or equal (unsigned 32-bit)
  amdgcn.cmpi s_cmp_le_u32 outs %scc ins %src0, %src1
    : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)

  return
}
