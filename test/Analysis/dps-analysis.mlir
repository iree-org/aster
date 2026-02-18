// RUN: aster-opt -test-dps-analysis %s 2>&1 | FileCheck %s
func.func private @rand() -> i1

// CHECK-LABEL:  function: "test_control_flow_provenance"
// CHECK:  Block: Block<op = func.func @test_control_flow_provenance() {...}, region = 0, bb = ^bb3, args = [%{{.*}}]>
// CHECK:    arguments: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  DPS analysis {
// CHECK:    value provenance {
// CHECK:      0 = `%{{.*}}` -> [2]
// CHECK:      2 = `%{{.*}}` -> [0]
// CHECK:      3 = `%{{.*}}` -> [1]
// CHECK:      4 = `%{{.*}}` -> [2]
// CHECK:    }
// CHECK:    control-flow provenance {
// CHECK:      0 = `%{{.*}}` -> {2 = `%{{.*}}`, 3 = `%{{.*}}`}
// CHECK:    }
// CHECK:  }
func.func @test_control_flow_provenance() {
  %0 = call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3(%1 : !amdgcn.vgpr)
^bb2:  // pred: ^bb0
  cf.br ^bb3(%2 : !amdgcn.vgpr)
^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
  %4 = amdgcn.test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  return
}

// CHECK-LABEL:  function: "test_value_provenance"
// CHECK:  Block: Block<op = func.func @test_value_provenance(%{{.*}}: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {...}, region = 0, bb = ^bb0, args = [%{{.*}}]>
// CHECK:    arguments: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr, !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}}:2 = amdgcn.test_inst outs %{{.*}}, %{{.*}} : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>)`
// CHECK:    results: [4 = `%{{.*}}#0`, 5 = `%{{.*}}#1`]
// CHECK:  Operation: `%{{.*}}:2 = amdgcn.split_register_range %{{.*}}#0 : !amdgcn.vgpr<[? + 2]>`
// CHECK:    results: [6 = `%{{.*}}#0`, 7 = `%{{.*}}#1`]
// CHECK:  Operation: `%{{.*}}:2 = amdgcn.split_register_range %{{.*}}#1 : !amdgcn.vgpr<[? + 2]>`
// CHECK:    results: [8 = `%{{.*}}#0`, 9 = `%{{.*}}#1`]
// CHECK:  DPS analysis {
// CHECK:    value provenance {
// CHECK:      0 = `%{{.*}}` -> [0, 1]
// CHECK:      1 = `%{{.*}}` -> [2]
// CHECK:      2 = `%{{.*}}` -> [3]
// CHECK:      3 = `%{{.*}}` -> [2, 3]
// CHECK:      4 = `%{{.*}}#0` -> [2, 3]
// CHECK:      5 = `%{{.*}}#1` -> [0, 1]
// CHECK:      6 = `%{{.*}}#0` -> [2]
// CHECK:      7 = `%{{.*}}#1` -> [3]
// CHECK:      8 = `%{{.*}}#0` -> [0]
// CHECK:      9 = `%{{.*}}#1` -> [1]
// CHECK:    }
// CHECK:    control-flow provenance {
// CHECK:    }
// CHECK:  }
func.func @test_value_provenance(%arg0: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3:2 = amdgcn.test_inst outs %2, %arg0 : (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>)
  %4:2 = amdgcn.split_register_range %3#0 : !amdgcn.vgpr<[? + 2]>
  %5:2 = amdgcn.split_register_range %3#1 : !amdgcn.vgpr<[? + 2]>
  return %4#0, %5#0 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:  function: "test_structured_control_flow_provenance"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!amdgcn.vgpr, !amdgcn.vgpr) {...}`
// CHECK:    results: [5 = `%{{.*}}#0`, 6 = `%{{.*}}#1`]
// CHECK:  Block: Block<op = %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!amdgcn.vgpr, !amdgcn.vgpr) {...}, region = 0, bb = ^bb0, args = [%{{.*}}, %{{.*}}, %{{.*}}]>
// CHECK:    arguments: [7 = `%{{.*}}`, 8 = `%{{.*}}`, 9 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr`
// CHECK:    results: [10 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}} : (!amdgcn.vgpr) -> !amdgcn.vgpr`
// CHECK:    results: [11 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}}:2 = amdgcn.test_inst outs %{{.*}}#0, %{{.*}}#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)`
// CHECK:    results: [12 = `%{{.*}}#0`, 13 = `%{{.*}}#1`]
// CHECK:  DPS analysis {
// CHECK:    value provenance {
// CHECK:      3 = `%{{.*}}` -> [0]
// CHECK:      4 = `%{{.*}}` -> [1]
// CHECK:      5 = `%{{.*}}#0` -> [4]
// CHECK:      6 = `%{{.*}}#1` -> [5]
// CHECK:      8 = `%{{.*}}` -> [2]
// CHECK:      9 = `%{{.*}}` -> [3]
// CHECK:      10 = `%{{.*}}` -> [2]
// CHECK:      11 = `%{{.*}}` -> [3]
// CHECK:      12 = `%{{.*}}#0` -> [4]
// CHECK:      13 = `%{{.*}}#1` -> [5]
// CHECK:    }
// CHECK:    control-flow provenance {
// CHECK:      5 = `%{{.*}}#0` -> {3 = `%{{.*}}`, 10 = `%{{.*}}`}
// CHECK:      6 = `%{{.*}}#1` -> {4 = `%{{.*}}`, 11 = `%{{.*}}`}
// CHECK:      8 = `%{{.*}}` -> {3 = `%{{.*}}`, 10 = `%{{.*}}`}
// CHECK:      9 = `%{{.*}}` -> {4 = `%{{.*}}`, 11 = `%{{.*}}`}
// CHECK:    }
// CHECK:  }
func.func @test_structured_control_flow_provenance() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2:2 = scf.for %arg0 = %c0 to %c10 step %c1 iter_args(%arg1 = %0, %arg2 = %1) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
    %4 = amdgcn.test_inst outs %arg1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %5 = amdgcn.test_inst outs %arg2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    scf.yield %4, %5 : !amdgcn.vgpr, !amdgcn.vgpr
  }
  %3:2 = amdgcn.test_inst outs %2#0, %2#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  return
}

// CHECK-LABEL:  function: "test_provenance_with_allocated_registers"
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<3>`
// CHECK:    results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr`
// CHECK:    results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.test_inst outs %{{.*}}, %{{.*}} : (!amdgcn.vgpr<3>, !amdgcn.vgpr) -> !amdgcn.vgpr`
// CHECK:    results: [2 = `%{{.*}}`]
// CHECK:  DPS analysis {
// CHECK:    value provenance {
// CHECK:      1 = `%{{.*}}` -> [0]
// CHECK:      2 = `%{{.*}}` -> [0]
// CHECK:    }
// CHECK:    control-flow provenance {
// CHECK:    }
// CHECK:  }
func.func @test_provenance_with_allocated_registers() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<3>
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.test_inst outs %0, %1 : (!amdgcn.vgpr<3>, !amdgcn.vgpr) -> !amdgcn.vgpr
  return
}
