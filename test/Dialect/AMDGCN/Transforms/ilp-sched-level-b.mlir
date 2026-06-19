// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=0 ilp-time-limit-ms=2000})))" | FileCheck %s --check-prefix=LEVEL0
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=1 ilp-time-limit-ms=2000})))" | FileCheck %s --check-prefix=LEVEL1

// Level-b VMEM interleaving: 9 mutually-independent global_load_dword ops
// followed by 4 VALU fillers.
// level 0: source-order biased; all 9 loads survive (no ops dropped).
// level 1: default vmem-gap=0 -> no VMEM spacing constraint -> identical to
//          level 0.  Both levels interleave loads and v_mov freely under the
//          source-order tie-break; the exact pattern is an ILP solver artifact.

!v   = !amdgcn.vgpr
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // All 9 loads must survive; exact order is solver-determined.
  // LEVEL0-LABEL:    kernel @ilp_level_b_vmem
  // LEVEL0-COUNT-9:  global_load_dword

  // Level 1 with vmem-gap=0 is identical to level 0.
  // LEVEL1-LABEL:    kernel @ilp_level_b_vmem
  // LEVEL1-COUNT-9:  global_load_dword
  amdgcn.kernel @ilp_level_b_vmem {
    %dst  = amdgcn.alloca : !v
    %sa0  = amdgcn.alloca : !s
    %sa1  = amdgcn.alloca : !s
    %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %off  = amdgcn.alloca : !v
    %c0i  = arith.constant 0 : i32
    // 9 mutually-independent VMEM loads, contiguous in source order. Tokens are
    // unused (no amdgcn.wait) so there are no serialization edges between them.
    %r0, %t0 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r1, %t1 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r2, %t2 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r3, %t3 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r4, %t4 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r5, %t5 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r6, %t6 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r7, %t7 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r8, %t8 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    // Independent VALU fillers (VMEM burst penalty = 0 for VALU). Their setup
    // (alloca + constants) supplies the non-VMEM ops the solver interleaves.
    %fa  = amdgcn.alloca : !v
    %cf0 = arith.constant 1 : i32
    %cf1 = arith.constant 2 : i32
    %cf2 = arith.constant 3 : i32
    %cf3 = arith.constant 4 : i32
    %fva = amdgcn.v_mov_b32 outs(%fa) ins(%cf0) : outs(!v) ins(i32)
    %fvb = amdgcn.v_mov_b32 outs(%fa) ins(%cf1) : outs(!v) ins(i32)
    %fvc = amdgcn.v_mov_b32 outs(%fa) ins(%cf2) : outs(!v) ins(i32)
    %fvd = amdgcn.v_mov_b32 outs(%fa) ins(%cf3) : outs(!v) ins(i32)
    amdgcn.end_kernel
  }
}
