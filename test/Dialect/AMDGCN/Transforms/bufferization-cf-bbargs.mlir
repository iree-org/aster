// RUN: aster-opt %s --amdgcn-bufferization --split-input-file | FileCheck %s

// Simple diamond CFG: two allocas merge at block argument.
// The pass should insert copies before each branch.
// CHECK-LABEL:     kernel @bufferization_phi_copies_1 {
//       CHECK:       %[[ALLOCA1:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       %[[ALLOCA2:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       cf.cond_br
//       CHECK:     ^bb1:
//       CHECK:       %[[COPY_ALLOC1:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       %[[COPY1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[COPY_ALLOC1]], %[[ALLOCA1]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:       cf.br ^bb3(%[[COPY1]] : !amdgcn.vgpr)
//       CHECK:     ^bb2:
//       CHECK:       %[[COPY_ALLOC2:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       %[[COPY2:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[COPY_ALLOC2]], %[[ALLOCA2]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:       cf.br ^bb3(%[[COPY2]] : !amdgcn.vgpr)
//       CHECK:     ^bb3(%[[ARG:.*]]: !amdgcn.vgpr):
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
amdgcn.module @bufferization_phi_copies_1 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_phi_copies_1 {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// Same alloca used in both branches - should NOT insert copies (only 1 alloca merges).
// CHECK-LABEL:   kernel @bufferization_single_alloca {
//       CHECK:     %[[ALLOCA:.*]] = alloca : !amdgcn.vgpr
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//   CHECK-NOT:       alloca
//   CHECK-NOT:       v_mov_b32_e32
//       CHECK:       cf.br ^bb3(%[[ALLOCA]] : !amdgcn.vgpr)
//       CHECK:     ^bb2:
//   CHECK-NOT:       alloca
//   CHECK-NOT:       v_mov_b32_e32
//       CHECK:       cf.br ^bb3(%[[ALLOCA]] : !amdgcn.vgpr)
//       CHECK:     ^bb3({{.*}}: !amdgcn.vgpr):
amdgcn.module @bufferization_single_alloca target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_single_alloca {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):
    test_inst ins %3 : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// SGPR type: should use s_mov_b32 instead of v_mov_b32_e32.
// CHECK-LABEL:   kernel @bufferization_sgpr_copies {
//       CHECK:     %[[ALLOCA1:.*]] = alloca : !amdgcn.sgpr
//       CHECK:     %[[ALLOCA2:.*]] = alloca : !amdgcn.sgpr
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//       CHECK:       %[[COPY_ALLOC1:.*]] = alloca : !amdgcn.sgpr
//       CHECK:       %[[COPY1:.*]] = sop1 s_mov_b32 outs %[[COPY_ALLOC1]] ins %[[ALLOCA1]] :
//       CHECK:       cf.br ^bb3(%[[COPY1]] : !amdgcn.sgpr)
//       CHECK:     ^bb2:
//       CHECK:       %[[COPY_ALLOC2:.*]] = alloca : !amdgcn.sgpr
//       CHECK:       %[[COPY2:.*]] = sop1 s_mov_b32 outs %[[COPY_ALLOC2]] ins %[[ALLOCA2]] :
//       CHECK:       cf.br ^bb3(%[[COPY2]] : !amdgcn.sgpr)
//       CHECK:     ^bb3({{.*}}: !amdgcn.sgpr):
amdgcn.module @bufferization_sgpr_copies target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_sgpr_copies {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.sgpr
    %2 = alloca : !amdgcn.sgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%1 : !amdgcn.sgpr)
  ^bb2:
    cf.br ^bb3(%2 : !amdgcn.sgpr)
  ^bb3(%3: !amdgcn.sgpr):
    test_inst ins %3 : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Values derived from allocas (not raw allocas) - should still insert copies.
// CHECK-LABEL:   kernel @bufferization_derived_values {
//       CHECK:     %[[ALLOCA1:.*]] = alloca : !amdgcn.vgpr
//       CHECK:     %[[ALLOCA2:.*]] = alloca : !amdgcn.vgpr
//       CHECK:     %[[V1:.*]] = test_inst outs %[[ALLOCA1]]
//       CHECK:     %[[V2:.*]] = test_inst outs %[[ALLOCA2]]
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//       CHECK:       %[[COPY_ALLOC1:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       %[[COPY1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[COPY_ALLOC1]], %[[V1]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:       cf.br ^bb3(%[[COPY1]] : !amdgcn.vgpr)
//       CHECK:     ^bb2:
//       CHECK:       %[[COPY_ALLOC2:.*]] = alloca : !amdgcn.vgpr
//       CHECK:       %[[COPY2:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[COPY_ALLOC2]], %[[V2]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:       cf.br ^bb3(%[[COPY2]] : !amdgcn.vgpr)
//       CHECK:     ^bb3({{.*}}: !amdgcn.vgpr):
amdgcn.module @bufferization_derived_values target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_derived_values {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %v1 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%v1 : !amdgcn.vgpr)
  ^bb2:
    cf.br ^bb3(%v2 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):
    test_inst ins %val : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}
