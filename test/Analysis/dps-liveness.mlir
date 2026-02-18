// RUN: aster-opt -test-dps-liveness %s 2>&1 | FileCheck %s

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
