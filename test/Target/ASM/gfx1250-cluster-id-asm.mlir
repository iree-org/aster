// RUN: aster-opt %s \
// RUN:   --pass-pipeline='builtin.module(amdgcn.module(amdgcn.kernel( \
// RUN:     aster-amdgcn-expand-md-ops, \
// RUN:     aster-amdgcn-bufferization, \
// RUN:     amdgcn-to-register-semantics)), \
// RUN:   amdgcn-backend, \
// RUN:   amdgcn-remove-test-inst, \
// RUN:   amdgcn-hazards{v_nops=0 s_nops=0})' \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
//
// RUN: aster-opt %s \
// RUN:   --pass-pipeline='builtin.module(amdgcn.module(amdgcn.kernel( \
// RUN:     aster-amdgcn-expand-md-ops, \
// RUN:     aster-amdgcn-bufferization, \
// RUN:     amdgcn-to-register-semantics)), \
// RUN:   amdgcn-backend, \
// RUN:   amdgcn-remove-test-inst, \
// RUN:   amdgcn-hazards{v_nops=0 s_nops=0})' \
// RUN: | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// CHECK-LABEL: cluster_ids:
//  CHECK: s_bfe_u32 s0, ttmp9, 2097152
//  CHECK: s_bfe_u32 s1, ttmp6, 262144
//  CHECK: s_bfe_u32 s2, ttmp6, 262156
//  CHECK: s_endpgm
amdgcn.module @m target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cluster_ids {
    %cx = amdgcn.cluster_id x : !amdgcn.sgpr
    %wx = amdgcn.cluster_workgroup_id x : !amdgcn.sgpr
    %mx = amdgcn.cluster_workgroup_max_id x : !amdgcn.sgpr
    amdgcn.test_inst ins %cx, %wx, %mx : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.end_kernel
  }
}
