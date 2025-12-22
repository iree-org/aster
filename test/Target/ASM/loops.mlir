// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Test 1: Simple Conditional (If-Then-Else) with trap
// CHECK:  ; Module: mod
// CHECK:  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK:  .text
// CHECK:  .globl test_if_then_else
// CHECK:  .p2align 8
// CHECK:  .type test_if_then_else,@function
// CHECK:test_if_then_else:
// CHECK:  s_mov_b32 s0, 5
// CHECK:  s_mov_b32 s1, 4
// CHECK:  s_cmp_le_i32 s0, s1
// CHECK:  s_cbranch_scc1 .AMDGCN_BB_1
// CHECK:  ; fallthrough: .AMDGCN_BB_2
// CHECK:.AMDGCN_BB_2:
// CHECK:  s_endpgm
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_trap 2
// CHECK:  s_endpgm

// Test 2: While Loop with Counter
// CHECK:  .text
// CHECK:  .globl test_while_loop
// CHECK:  .p2align 8
// CHECK:  .type test_while_loop,@function
// CHECK:test_while_loop:
// CHECK:  s_mov_b32 s2, 10
// CHECK:  s_mov_b32 s3, 9
// CHECK:  ; fallthrough: .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_1:
// CHECK:  s_cmp_lt_i32 s3, s2
// CHECK:  s_cbranch_scc0 .AMDGCN_BB_2
// CHECK:  ; fallthrough: .AMDGCN_BB_3
// CHECK:.AMDGCN_BB_3:
// CHECK:  s_branch .AMDGCN_BB_1
// CHECK:.AMDGCN_BB_2:
// CHECK:  s_endpgm

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Test 1: Simple Conditional (If-Then-Else) with trap
  // Pattern: if (5 <= 4) { trap(); } else { end; }
  amdgcn.kernel @test_if_then_else {
  ^entry:
    %c5 = arith.constant 5 : i32
    %c4 = arith.constant 4 : i32
    %scc = amdgcn.alloca : !amdgcn.sreg<scc, 0>
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>

    %r0 = amdgcn.sop1 s_mov_b32 outs %s0 ins %c5 : !amdgcn.sgpr<0>, i32
    %r1 = amdgcn.sop1 s_mov_b32 outs %s1 ins %c4 : !amdgcn.sgpr<1>, i32
    %scc_result = amdgcn.sopc s_cmp_le_i32 outs %scc ins %r0, %r1
      : !amdgcn.sreg<scc, 0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>

    amdgcn.cbranch #amdgcn.inst<s_cbranch_scc1> %scc_result ^then fallthrough (^else)
      : !amdgcn.sreg<scc, 0>
  ^else:
    amdgcn.end_kernel
  ^then:
    amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 2
    amdgcn.end_kernel
  }

  // Test 2: While Loop with Counter
  // Pattern: counter=9; limit=10; while (counter < limit) { }
  amdgcn.kernel @test_while_loop {
  ^entry:
    %c10 = arith.constant 10 : i32
    %c9 = arith.constant 9 : i32
    %scc = amdgcn.alloca : !amdgcn.sreg<scc, 0>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>

    %r2 = amdgcn.sop1 s_mov_b32 outs %s2 ins %c10 : !amdgcn.sgpr<2>, i32
    %r3 = amdgcn.sop1 s_mov_b32 outs %s3 ins %c9 : !amdgcn.sgpr<3>, i32
    amdgcn.branch #amdgcn.inst<s_branch> ^loop_header

  ^loop_header:
    %scc_result = amdgcn.sopc s_cmp_lt_i32 outs %scc ins %r3, %r2
      : !amdgcn.sreg<scc, 0>, !amdgcn.sgpr<3>, !amdgcn.sgpr<2>
    amdgcn.cbranch #amdgcn.inst<s_cbranch_scc0> %scc_result ^exit fallthrough (^loop_body)
      : !amdgcn.sreg<scc, 0>
  ^loop_body:
    amdgcn.branch #amdgcn.inst<s_branch> ^loop_header
  ^exit:
    amdgcn.end_kernel
  }
}
