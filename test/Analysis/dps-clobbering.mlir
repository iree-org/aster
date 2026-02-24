// RUN: aster-opt -test-dps-clobbering --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: @test_no_clobbering
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [false]
func.func @test_no_clobbering() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr) -> ()
  return
}

// -----

func.func private @rand() -> i1
// CHECK-LABEL: @test_control_flow_liveness
// CHECK-NOT: [{{true|false}}
func.func @test_control_flow_liveness() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.test_inst ins %2 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3(%1 : !amdgcn.vgpr)
^bb2:  // pred: ^bb0
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3(%2 : !amdgcn.vgpr)
^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
  amdgcn.test_inst ins %3, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_range_liveness
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
// CHECK:      [true]
func.func @test_range_liveness() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %2 : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
  amdgcn.test_inst ins %3, %0 : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_split_range_liveness
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
func.func @test_split_range_liveness() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %2 : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
  %4, %5 = amdgcn.split_register_range %2 : !amdgcn.vgpr<[? + 2]>
  %6 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  %7 = amdgcn.alloca : !amdgcn.vgpr
  amdgcn.test_inst ins %7, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_range_with_intermediate_simultaneously_live
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
func.func @test_range_with_intermediate_simultaneously_live() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %4 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %5 = amdgcn.test_inst outs %2 ins %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  %6 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  amdgcn.test_inst ins %6, %2 : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_range_intermediate_used_before_range
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
func.func @test_range_intermediate_used_before_range() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %4 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %5 = amdgcn.test_inst outs %2 ins %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %2 : (!amdgcn.vgpr) -> ()
  %6 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  amdgcn.test_inst ins %6 : (!amdgcn.vgpr<[? + 2]>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_range_intermediate_used_after_range
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [true]
func.func @test_range_intermediate_used_after_range() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %4 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %5 = amdgcn.test_inst outs %2 ins %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  %6 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  amdgcn.test_inst ins %6 : (!amdgcn.vgpr<[? + 2]>) -> ()
  amdgcn.test_inst ins %2 : (!amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_partial_split_liveness
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
// CHECK:      [true]
func.func @test_partial_split_liveness() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %2 : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
  %4:2 = amdgcn.split_register_range %2 : !amdgcn.vgpr<[? + 2]>
  amdgcn.test_inst ins %4#0 : (!amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_reg_interference_no_liveness_effect
// CHECK-NOT: [{{true|false}}
func.func @test_reg_interference_no_liveness_effect() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  amdgcn.test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  amdgcn.reg_interference %0, %2 : !amdgcn.vgpr, !amdgcn.vgpr
  amdgcn.test_inst ins %2 : (!amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_scf_for
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [false]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [false]
func.func @test_scf_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2:2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0, %arg2 = %1) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
    %3 = amdgcn.test_inst outs %arg1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %4 = amdgcn.test_inst outs %arg2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %3, %4 : !amdgcn.vgpr, !amdgcn.vgpr
  }
  amdgcn.test_inst ins %2#0, %2#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  amdgcn.test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_scf_for_ping_pong
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [false]
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:      [false]
func.func @test_scf_for_ping_pong() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2:2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0, %arg2 = %1) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
    %3 = amdgcn.test_inst outs %arg1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %4 = amdgcn.test_inst outs %arg2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // Ping-pong: swap iter_args
    scf.yield %4, %3 : !amdgcn.vgpr, !amdgcn.vgpr
  }
  amdgcn.test_inst ins %2#0, %2#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  amdgcn.test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_multi_results
// CHECK:    %{{.*}}:2 = amdgcn.test_inst outs %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:      [true, true]
func.func @test_multi_results() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2, %3 = amdgcn.test_inst outs %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  amdgcn.test_inst ins %0, %1, %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL: @test_multi_results_asymmetric
// CHECK:    %{{.*}}:2 = amdgcn.test_inst outs %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:      [true, false]
func.func @test_multi_results_asymmetric() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2, %3 = amdgcn.test_inst outs %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  amdgcn.test_inst ins %0 : (!amdgcn.vgpr) -> ()
  return
}
