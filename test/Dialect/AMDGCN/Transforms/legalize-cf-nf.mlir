// RUN: aster-opt --pass-pipeline='builtin.module(any(amdgcn-legalize-cf))' %s \
// RUN:   | FileCheck %s

// Verify that legalize-cf sets the no_cf_branches post-condition.

// CHECK-LABEL: kernel @sets_postcondition
// CHECK-SAME: attributes {normal_forms = [#amdgcn.all_registers_allocated, #amdgcn.no_cf_branches]}
amdgcn.kernel @sets_postcondition attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
^bb0:
  %c0_i32 = arith.constant 0 : i32
  %c10_i32 = arith.constant 10 : i32
  %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
  %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
  amdgcn.s_mov_b32 outs(%alloc0) ins(%c0_i32) : outs(!amdgcn.sgpr<0>) ins(i32)
  amdgcn.s_mov_b32 outs(%alloc1) ins(%c10_i32) : outs(!amdgcn.sgpr<1>) ins(i32)
  %scc_cmp = amdgcn.alloca : !amdgcn.scc<0>
  lsir.cmpi i32 slt %scc_cmp, %alloc0, %alloc1 : !amdgcn.scc<0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  lsir.cond_br %scc_cmp : !amdgcn.scc<0>, ^bb1, ^bb2
^bb1:
  amdgcn.end_kernel
^bb2:
  amdgcn.end_kernel
}
