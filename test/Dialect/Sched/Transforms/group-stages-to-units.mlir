// RUN: aster-opt %s --sched-group-stages-to-units | FileCheck %s

// CHECK-LABEL: func.func @test_group_two_stages
// CHECK: scf.for
// CHECK:   sched.unit
// CHECK-SAME: stage 0
// CHECK:   sched.unit
// CHECK-SAME: stage 1
func.func @test_group_two_stages(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %a = arith.addi %arg0, %arg0 {sched.stage = 0 : i32} : i32
    %b = arith.muli %a, %arg0 {sched.stage = 1 : i32} : i32
  }
  return
}

// CHECK-LABEL: func.func @test_group_ssa_dep_assignment
// CHECK: scf.for
// CHECK:   sched.unit
// CHECK-SAME: stage 0
// The constant has no stage attr but is consumed by stage-0 op.
func.func @test_group_ssa_dep_assignment(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %cst = arith.constant 1 : i32
    %a = arith.addi %arg0, %cst {sched.stage = 0 : i32} : i32
  }
  return
}

// Verify that multiple ops with the same stage number are grouped into one unit.
// CHECK-LABEL: func.func @test_duplicate_stage_merged
// CHECK: scf.for
// CHECK:   sched.unit
// CHECK-SAME: stage 0
// CHECK-NOT:  sched.unit
func.func @test_duplicate_stage_merged(%arg0: i32) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %a = arith.addi %arg0, %arg0 {sched.stage = 0 : i32} : i32
    %b = arith.muli %arg0, %arg0 {sched.stage = 0 : i32} : i32
  }
  return
}
