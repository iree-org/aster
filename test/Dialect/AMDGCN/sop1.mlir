// RUN: aster-opt %s --verify-roundtrip

// Test SOP1 (Scalar Operations with 1 operand) instructions
// SOP1 instructions perform scalar operations with one source operand

func.func @sop1_mov_register(%sdst: !amdgcn.sgpr, %src: !amdgcn.sgpr) -> !amdgcn.sgpr {
  // s_mov_b32 - SOP1 move 32-bit value from register
  %result = amdgcn.sop1 s_mov_b32 outs %sdst ins %src
    : !amdgcn.sgpr, !amdgcn.sgpr
  return %result : !amdgcn.sgpr
}

func.func @sop1_mov_immediate(%sdst: !amdgcn.sgpr) -> !amdgcn.sgpr {
  // s_mov_b32 - SOP1 move 32-bit immediate value
  %imm = arith.constant 42 : i32
  %result = amdgcn.sop1 s_mov_b32 outs %sdst ins %imm
    : !amdgcn.sgpr, i32
  return %result : !amdgcn.sgpr
}

func.func @sop1_mov_to_m0(%m0: !amdgcn.m0, %src: !amdgcn.sgpr) {
  // s_mov_b32 - SOP1 move SGPR value to M0 register.
  // M0 has Allocated semantics (fixed physical register), so no SSA result.
  amdgcn.sop1 s_mov_b32 outs %m0 ins %src
    : !amdgcn.m0, !amdgcn.sgpr
  return
}

func.func @sop1_mov_imm_to_m0(%m0: !amdgcn.m0) {
  // s_mov_b32 - SOP1 move immediate to M0 register
  %imm = arith.constant 1024 : i32
  amdgcn.sop1 s_mov_b32 outs %m0 ins %imm
    : !amdgcn.m0, i32
  return
}

func.func @sop1_mov_m0_to_sgpr(%sdst: !amdgcn.sgpr, %m0: !amdgcn.m0) -> !amdgcn.sgpr {
  // s_mov_b32 - SOP1 read M0 back into an SGPR
  %result = amdgcn.sop1 s_mov_b32 outs %sdst ins %m0
    : !amdgcn.sgpr, !amdgcn.m0
  return %result : !amdgcn.sgpr
}
