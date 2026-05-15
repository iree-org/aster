// RUN: aster-opt --aster-scf-pipeline -split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @two_stage_nested_consumer

// Prologue: stage-0 scf.if executes unconditionally before the kernel loop.
// CHECK:       %[[PRO:.*]] = scf.if
// CHECK:         amdgcn.test_inst

// Kernel: stage-0 result becomes the single cross-stage iter_arg.
// CHECK:       %[[KER:.*]] = scf.for %{{.*}} iter_args(%[[CSV:.*]] = %[[PRO]]) -> (!amdgcn.vgpr)

// Kernel stage 0: independent scf.if, no cross-stage consumer.
// CHECK:         scf.if
// CHECK:           amdgcn.test_inst

// Kernel stage 1: scf.if body consumes the iter_arg -- the bug-fix invariant.
// CHECK:         scf.if
// CHECK:           affine.apply
// CHECK:           amdgcn.test_inst outs %{{.*}} ins %[[CSV]]

// CHECK:         scf.yield

// Epilogue: stage-1 scf.if drains using the for's scalar result.
// CHECK:       scf.if
// CHECK:         affine.apply
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[KER]]
// CHECK:       return

func.func @two_stage_nested_consumer(%ub: index, %cond: i1) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = scf.if %cond -> (!amdgcn.vgpr) {
      %t = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
      scf.yield %t : !amdgcn.vgpr
    } else {
      scf.yield %s0 : !amdgcn.vgpr
    } {sched.stage = 0 : i32}
    %w = scf.if %cond -> (!amdgcn.vgpr) {
      %off = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %t = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
      scf.yield %t : !amdgcn.vgpr
    } else {
      scf.yield %s1 : !amdgcn.vgpr
    } {sched.stage = 1 : i32}
  }
  return
}

// -----

// CHECK-LABEL: func.func @scfif_result_to_flat_consumer
// CHECK:       %[[PRO:.*]] = scf.if
// CHECK:       %[[KER:.*]] = scf.for %{{.*}} iter_args(%[[CSV:.*]] = %[[PRO]])
// CHECK:         amdgcn.test_inst outs %{{.*}} ins %[[CSV]]
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %[[KER]]

func.func @scfif_result_to_flat_consumer(%ub: index, %cond: i1) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = scf.if %cond -> (!amdgcn.vgpr) {
      %t = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
      scf.yield %t : !amdgcn.vgpr
    } else {
      scf.yield %s0 : !amdgcn.vgpr
    } {sched.stage = 0 : i32}
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----

func.func @scfif_stage_divergent_nested(%ub: index, %cond: i1) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = scf.if %cond -> (!amdgcn.vgpr) {
      // expected-error @below {{aster-scf-pipeline: scf.if used as a pipeline stage has a nested op with divergent sched.stage}}
      %t = amdgcn.test_inst outs %s0 {sched.stage = 1 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
      scf.yield %t : !amdgcn.vgpr
    } else {
      scf.yield %s0 : !amdgcn.vgpr
    } {sched.stage = 0 : i32}
    %w = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----

func.func @region_stage_non_scfif(%ub: index) {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    %v = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // expected-error @below {{aster-scf-pipeline: region-carrying op used as a pipeline stage is supported only for scf.if}}
    %r = scf.execute_region -> (!amdgcn.vgpr) {
      %t = amdgcn.test_inst outs %s1 ins %v {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
      scf.yield %t : !amdgcn.vgpr
    } {sched.stage = 1 : i32}
  }
  return
}
