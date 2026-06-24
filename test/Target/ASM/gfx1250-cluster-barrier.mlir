// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
//
// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// Cluster-scope token_barrier lowers to the gfx1250 split cluster barrier.

// CHECK-LABEL: cbar:
//  CHECK: s_barrier_signal_isfirst -1
//  CHECK: s_barrier_wait -1
//  CHECK: s_cbranch_scc0 [[JOIN:.*]]
//  CHECK: s_barrier_signal -3
//  CHECK: s_branch [[JOIN]]
//  CHECK: s_barrier_wait -3
//  CHECK: s_endpgm

amdgcn.module @cbar_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cbar attributes {cluster_dims = array<i32: 2, 1, 1>} {
    %fence = amdgcn.token_barrier scope(#amdgcn.barrier_scope<cluster>)
    amdgcn.end_kernel
  }
}
