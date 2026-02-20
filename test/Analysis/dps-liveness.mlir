// RUN: aster-opt -test-dps-liveness --split-input-file %s 2>&1 | FileCheck %s

func.func private @rand() -> i1

// CHECK-LABEL:  function: "test_control_flow_liveness"
// CHECK:  Block: Block<op = func.func @test_control_flow_liveness() {...}, region = 0, bb = ^bb3, args = [%{{.*}}]>
// CHECK:    arguments: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr) -> () -> {2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr) -> () -> {2 = `%{{.*}}`, 3 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_range_liveness"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr, !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> -> {0 = `%{{.*}}`, 1 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
func.func @test_range_liveness() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %2 : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>
  amdgcn.test_inst ins %3, %0 : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> ()
  return
}

// -----

// CHECK-LABEL:  function: "test_split_range_liveness"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr, !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}}:2 = amdgcn.split_register_range %{{.*}} : !amdgcn.vgpr<[? + 2]>`
// CHECK:    results: [4 = `%{{.*}}#0`, 5 = `%{{.*}}#1`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr`
// CHECK:    results: [6 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [7 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> -> {0 = `%{{.*}}`, 1 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr -> {0 = `%{{.*}}`, 1 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}#0, %{{.*}}#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {1 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_range_with_intermediate_simultaneously_live"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_range_intermediate_used_before_range"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr) -> () -> {0 = `%{{.*}}`, 1 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_range_intermediate_used_after_range"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs %{{.*}} ins {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr<[? + 2]>) -> () -> {2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_partial_split_liveness"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {0 = `%{{.*}}`, 1 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}#0 : (!amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_reg_interference_no_liveness_effect"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    amdgcn.test_inst ins {{.*}} -> {2 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}} : (!amdgcn.vgpr) -> () -> {}
// CHECK:  }
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

// CHECK-LABEL:  function: "test_scf_for_liveness"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {3 = `%{{.*}}`, 4 = `%{{.*}}`, 8 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {3 = `%{{.*}}`, 4 = `%{{.*}}`, 8 = `%{{.*}}`, 9 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}#0, %{{.*}}#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {3 = `%{{.*}}`, 4 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
func.func @test_scf_for_liveness() {
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

// CHECK-LABEL:  function: "test_scf_for_ping_pong_liveness"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  DPS liveness (after program points) {
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {3 = `%{{.*}}`, 8 = `%{{.*}}`}
// CHECK:    %{{.*}} = amdgcn.test_inst outs {{.*}} -> {3 = `%{{.*}}`, 8 = `%{{.*}}`, 9 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}#0, %{{.*}}#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {3 = `%{{.*}}`, 4 = `%{{.*}}`}
// CHECK:    amdgcn.test_inst ins %{{.*}}, %{{.*}} : (!amdgcn.vgpr, !amdgcn.vgpr) -> () -> {}
// CHECK:  }
func.func @test_scf_for_ping_pong_liveness() {
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
