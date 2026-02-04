// RUN: aster-opt %s --cse --amdgcn-register-allocation --cse --aster-disable-verifiers --aster-suppress-disabled-verifier-warning --split-input-file | FileCheck %s

// Simple diamond CFG: two allocas merge at block argument.
// The pass inserts copies and coalesces them to the same register.
// CHECK-LABEL:     kernel @ra_phi_coalescing_1 {
//       CHECK:       cf.cond_br
//       CHECK:     ^bb1:
//   CHECK-NOT:       alloc
//   CHECK-NOT:       mov
//       CHECK:       cf.br ^bb3(%[[B0:.*]] : !amdgcn.vgpr<0>)
//       CHECK:     ^bb2:
//   CHECK-NOT:       alloc
//       CHECK:       %[[B1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}} : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
//       CHECK:       cf.br ^bb3(%[[B1:.*]] : !amdgcn.vgpr<0>)
//       CHECK:     ^bb3(%[[B:.*]]: !amdgcn.vgpr<0>):
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
amdgcn.module @ra_phi_coalescing_1 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @ra_phi_coalescing_1 {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %3 = alloca : !amdgcn.vgpr
    %4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %3, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%4 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    %5 = alloca : !amdgcn.vgpr
    %6 = amdgcn.vop1.vop1 <v_mov_b32_e32> %5, %2 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%6 : !amdgcn.vgpr)
  ^bb3(%7: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    %8 = test_inst outs %7 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK-LABEL:   kernel @ra_phi_coalescing_2 {
//       CHECK:     lsir.cmpi i32 eq
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//   CHECK-NOT:       alloc
//   CHECK-NOT:       mov
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<0>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
//       CHECK:       %[[B0:.*]] = test_inst {{.*}} : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
//       CHECK:       cf.br ^bb3(%[[B0:.*]] : !amdgcn.vgpr<0>)
//       CHECK:     ^bb2:
//   CHECK-NOT:       alloc
//   CHECK-NOT:       mov
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
//       CHECK:       %[[B1:.*]] = test_inst {{.*}} : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
//       CHECK:       cf.br ^bb3(%[[B1:.*]] : !amdgcn.vgpr<0>)
//       CHECK:     ^bb3(%[[B:.*]]: !amdgcn.vgpr<0>):
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<0>) -> ()
amdgcn.module @ra_phi_coalescing_2 target = <gfx942> isa = <cdna3> {
  kernel @ra_phi_coalescing_2 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0_i32 = arith.constant 0 : i32
    %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr, i32
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = test_inst outs %4 ins %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %10 = alloca : !amdgcn.vgpr
    %11 = test_inst outs %10 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %12 = alloca : !amdgcn.vgpr
    %13 = amdgcn.vop1.vop1 <v_mov_b32_e32> %12, %11 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%13 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    %14 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %15 = alloca : !amdgcn.vgpr
    %16 = test_inst outs %15 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %17 = alloca : !amdgcn.vgpr
    %18 = amdgcn.vop1.vop1 <v_mov_b32_e32> %17, %16 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%18 : !amdgcn.vgpr)
  ^bb3(%19: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %19 : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   kernel @ra_phi_coalescing_3 {
//       CHECK:     lsir.cmpi i32 eq
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//   CHECK-NOT:       alloc
//   CHECK-NOT:       mov
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
//       CHECK:       %[[B0:.*]] = test_inst {{.*}} : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
//       CHECK:       cf.br ^bb3(%[[B0:.*]] : !amdgcn.vgpr<1>)
//       CHECK:     ^bb2:
//   CHECK-NOT:       alloc
//   CHECK-NOT:       mov
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
//       CHECK:       %[[B1:.*]] = test_inst {{.*}} : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
//       CHECK:       cf.br ^bb3(%[[B1:.*]] : !amdgcn.vgpr<1>)
//       CHECK:     ^bb3(%[[B:.*]]: !amdgcn.vgpr<1>):
//       CHECK:       test_inst {{.*}} : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> ()
amdgcn.module @ra_phi_coalescing_3 target = <gfx942> isa = <cdna3> {
  kernel @ra_phi_coalescing_3 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0_i32 = arith.constant 0 : i32
    %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr, i32
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = test_inst outs %4 ins %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %10 = alloca : !amdgcn.vgpr
    %11 = test_inst outs %10 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %12 = alloca : !amdgcn.vgpr
    %13 = amdgcn.vop1.vop1 <v_mov_b32_e32> %12, %11 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%13 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    %14 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %15 = alloca : !amdgcn.vgpr
    %16 = test_inst outs %15 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %17 = alloca : !amdgcn.vgpr
    %18 = amdgcn.vop1.vop1 <v_mov_b32_e32> %17, %16 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%18 : !amdgcn.vgpr)
  ^bb3(%19: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %19, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   kernel @ra_phi_coalescing_4 {
//       CHECK:     lsir.cmpi i32 eq
//       CHECK:     cf.cond_br
//       CHECK:     ^bb1:
//       CHECK:       %[[ALLOC0:.*]] = alloca : !amdgcn.vgpr<2>
//       CHECK:       amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}} : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<2>
//       CHECK:       cf.br ^bb3({{.*}} : !amdgcn.vgpr<2>)
//       CHECK:     ^bb2:
//       CHECK:       %[[ALLOC1:.*]] = alloca : !amdgcn.vgpr<2>
//       CHECK:       amdgcn.vop1.vop1 <v_mov_b32_e32> {{.*}} : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
//       CHECK:       cf.br ^bb3({{.*}} : !amdgcn.vgpr<2>)
//       CHECK:     ^bb3({{.*}}: !amdgcn.vgpr<2>):
amdgcn.module @ra_phi_coalescing_4 target = <gfx942> isa = <cdna3> {
  kernel @ra_phi_coalescing_4 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0_i32 = arith.constant 0 : i32
    %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr, i32
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = alloca : !amdgcn.vgpr
    %10 = amdgcn.vop1.vop1 <v_mov_b32_e32> %9, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%10 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    %11 = alloca : !amdgcn.vgpr
    %12 = amdgcn.vop1.vop1 <v_mov_b32_e32> %11, %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%12 : !amdgcn.vgpr)
  ^bb3(%13: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %13, %6, %7 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}
