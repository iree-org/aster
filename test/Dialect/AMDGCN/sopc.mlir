// RUN: aster-opt %s --verify-roundtrip

// Test SOPC (Scalar Operations Compare) instructions
// SOPC instructions compare two scalar values and set the SCC register

func.func @sopc_signed_comparisons(%src0: !amdgcn.sgpr, %src1: !amdgcn.sgpr, %scc: !amdgcn.sreg<scc>)
    -> (!amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>) {
  // s_cmp_eq_i32 - SOPC compare equal (signed 32-bit)
  %scc_eq = amdgcn.sopc s_cmp_eq_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_lg_i32 - SOPC compare not equal (signed 32-bit)
  %scc_lg = amdgcn.sopc s_cmp_lg_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_gt_i32 - SOPC compare greater than (signed 32-bit)
  %scc_gt = amdgcn.sopc s_cmp_gt_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_ge_i32 - SOPC compare greater than or equal (signed 32-bit)
  %scc_ge = amdgcn.sopc s_cmp_ge_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_lt_i32 - SOPC compare less than (signed 32-bit)
  %scc_lt = amdgcn.sopc s_cmp_lt_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_le_i32 - SOPC compare less than or equal (signed 32-bit)
  %scc_le = amdgcn.sopc s_cmp_le_i32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  return %scc_eq, %scc_lg, %scc_gt, %scc_ge, %scc_lt, %scc_le
    : !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>
}

func.func @sopc_unsigned_comparisons(%src0: !amdgcn.sgpr, %src1: !amdgcn.sgpr, %scc: !amdgcn.sreg<scc>)
    -> (!amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>) {
  // s_cmp_eq_u32 - SOPC compare equal (unsigned 32-bit)
  %scc_eq = amdgcn.sopc s_cmp_eq_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_lg_u32 - SOPC compare not equal (unsigned 32-bit)
  %scc_lg = amdgcn.sopc s_cmp_lg_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_gt_u32 - SOPC compare greater than (unsigned 32-bit)
  %scc_gt = amdgcn.sopc s_cmp_gt_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_ge_u32 - SOPC compare greater than or equal (unsigned 32-bit)
  %scc_ge = amdgcn.sopc s_cmp_ge_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_lt_u32 - SOPC compare less than (unsigned 32-bit)
  %scc_lt = amdgcn.sopc s_cmp_lt_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  // s_cmp_le_u32 - SOPC compare less than or equal (unsigned 32-bit)
  %scc_le = amdgcn.sopc s_cmp_le_u32 outs %scc ins %src0, %src1
    : !amdgcn.sreg<scc>, !amdgcn.sgpr, !amdgcn.sgpr

  return %scc_eq, %scc_lg, %scc_gt, %scc_ge, %scc_lt, %scc_le
    : !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>, !amdgcn.sreg<scc>
}
