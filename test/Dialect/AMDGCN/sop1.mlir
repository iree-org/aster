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
