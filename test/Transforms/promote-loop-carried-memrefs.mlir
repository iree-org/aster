// RUN: aster-opt %s -aster-promote-loop-carried-memrefs -allow-unregistered-dialect --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @single_alloca
// CHECK-NOT: memref.alloca
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT:.*]]) -> (f32)
// CHECK:   %[[NEW:.*]] = arith.addf %[[ACC]], %[[ACC]]
// CHECK:   scf.yield %[[NEW]]
// CHECK: return %[[RESULT]]
func.func @single_alloca(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[] : memref<f32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[] : memref<f32>
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// CHECK-LABEL: func.func @multiple_allocas
// CHECK-NOT: memref.alloca
// CHECK: %[[RESULTS:.*]]:2 = scf.for {{.*}} iter_args(%[[ACC0:.*]] = %[[INIT0:.*]], %[[ACC1:.*]] = %[[INIT1:.*]]) -> (f32, f32)
// CHECK:   %[[NEW0:.*]] = arith.addf %[[ACC0]], %[[ACC0]]
// CHECK:   %[[NEW1:.*]] = arith.mulf %[[ACC1]], %[[ACC1]]
// CHECK:   scf.yield %[[NEW0]], %[[NEW1]]
// CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1
func.func @multiple_allocas(%init0: f32, %init1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %buf0 = memref.alloca() : memref<f32>
  %buf1 = memref.alloca() : memref<f32>
  memref.store %init0, %buf0[] : memref<f32>
  memref.store %init1, %buf1[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old0 = memref.load %buf0[] : memref<f32>
    %new0 = arith.addf %old0, %old0 : f32
    memref.store %new0, %buf0[] : memref<f32>
    %old1 = memref.load %buf1[] : memref<f32>
    %new1 = arith.mulf %old1, %old1 : f32
    memref.store %new1, %buf1[] : memref<f32>
  }
  %final0 = memref.load %buf0[] : memref<f32>
  %final1 = memref.load %buf1[] : memref<f32>
  return %final0, %final1 : f32, f32
}

// -----

// CHECK-LABEL: func.func @preserves_existing_iter_args
// CHECK: scf.for {{.*}} iter_args(%[[EXISTING:.*]] = %[[EXIST_INIT:.*]], %[[ACC:.*]] = %[[ALLOCA_INIT:.*]]) -> (f32, f32)
// CHECK:   %[[NEW_EXIST:.*]] = arith.addf %[[EXISTING]], %[[EXISTING]]
// CHECK:   %[[NEW_ACC:.*]] = arith.mulf %[[ACC]], %[[ACC]]
// CHECK:   scf.yield %[[NEW_EXIST]], %[[NEW_ACC]]
func.func @preserves_existing_iter_args(%init: f32, %init2: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %buf = memref.alloca() : memref<f32>
  memref.store %init2, %buf[] : memref<f32>
  %existing_result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %new = arith.addf %acc, %acc : f32
    %old = memref.load %buf[] : memref<f32>
    %new2 = arith.mulf %old, %old : f32
    memref.store %new2, %buf[] : memref<f32>
    scf.yield %new : f32
  }
  %final = memref.load %buf[] : memref<f32>
  return %existing_result, %final : f32, f32
}

// -----

// Negative test: alloca used outside the loop-carried pattern should be skipped.
// CHECK-LABEL: func.func @negative_no_loop
// CHECK: memref.alloca
func.func @negative_no_loop(%val: f32) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %val, %buf[] : memref<f32>
  %result = memref.load %buf[] : memref<f32>
  return %result : f32
}

// -----

// Negative test: multi-element memref should be skipped.
// CHECK-LABEL: func.func @negative_multi_element
// CHECK: memref.alloca
func.func @negative_multi_element(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %c0 = arith.constant 0 : index
  %buf = memref.alloca() : memref<4xf32>
  memref.store %init, %buf[%c0] : memref<4xf32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[%c0] : memref<4xf32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[%c0] : memref<4xf32>
  }
  %final = memref.load %buf[%c0] : memref<4xf32>
  return %final : f32
}

// -----

// Test with 1-d memref<1xT> (should still be promoted as scalar).
// CHECK-LABEL: func.func @one_element_1d
// CHECK-NOT: memref.alloca
// CHECK: scf.for {{.*}} iter_args
func.func @one_element_1d(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %c0 = arith.constant 0 : index
  %buf = memref.alloca() : memref<1xf32>
  memref.store %init, %buf[%c0] : memref<1xf32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[%c0] : memref<1xf32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[%c0] : memref<1xf32>
  }
  %final = memref.load %buf[%c0] : memref<1xf32>
  return %final : f32
}

// -----

// Negative: store-before-load in loop body. The load reads the value written
// in the SAME iteration. Promoting to iter_arg would incorrectly give the
// load the PREVIOUS iteration's value, violating the memory dependency.
// CHECK-LABEL: func.func @negative_store_before_load
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK-NOT: iter_args
func.func @negative_store_before_load(%init: f32, %x: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %new = arith.addf %x, %x : f32
    memref.store %new, %buf[] : memref<f32>
    %val = memref.load %buf[] : memref<f32>
    "test.use"(%val) : (f32) -> ()
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Negative: two stores in the loop. Ambiguous which value carries across
// iterations -- the first store's value is overwritten by the second.
// CHECK-LABEL: func.func @negative_two_stores_in_loop
// CHECK: memref.alloca
func.func @negative_two_stores_in_loop(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[] : memref<f32>
    %mid = arith.addf %old, %old : f32
    memref.store %mid, %buf[] : memref<f32>
    %new = arith.mulf %mid, %mid : f32
    memref.store %new, %buf[] : memref<f32>
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Negative: two loads in the loop. The alloca is read at two points where the
// value may differ (one before the store, one after). A single iter_arg cannot
// represent both.
// CHECK-LABEL: func.func @negative_two_loads_in_loop
// CHECK: memref.alloca
func.func @negative_two_loads_in_loop(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[] : memref<f32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[] : memref<f32>
    %readback = memref.load %buf[] : memref<f32>
    "test.use"(%readback) : (f32) -> ()
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Negative: extra store between loop and final load. The final load observes
// the intervening store, not the loop's result.
// CHECK-LABEL: func.func @negative_extra_post_loop_store
// CHECK: memref.alloca
func.func @negative_extra_post_loop_store(%init: f32, %other: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[] : memref<f32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[] : memref<f32>
  }
  memref.store %other, %buf[] : memref<f32>
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Negative: nested loops where init store and final load are in a different
// block from the enclosing loop. The pass requires all pattern ops to be
// siblings in the same block.
// CHECK-LABEL: func.func @negative_nested_different_blocks
// CHECK: memref.alloca
func.func @negative_nested_different_blocks(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  memref.store %init, %buf[] : memref<f32>
  scf.for %outer = %c0 to %c10 step %c1 {
    scf.for %i = %lb to %ub step %step {
      %old = memref.load %buf[] : memref<f32>
      %new = arith.addf %old, %old : f32
      memref.store %new, %buf[] : memref<f32>
    }
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Two sibling loops that each qualify independently with their own alloca.
// Both should be promoted.
// CHECK-LABEL: func.func @sibling_loops
// CHECK-NOT: memref.alloca
// CHECK: %[[R0:.*]] = scf.for {{.*}} iter_args
// CHECK: %[[R1:.*]] = scf.for {{.*}} iter_args
func.func @sibling_loops(%init0: f32, %init1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %buf0 = memref.alloca() : memref<f32>
  memref.store %init0, %buf0[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf0[] : memref<f32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf0[] : memref<f32>
  }
  %final0 = memref.load %buf0[] : memref<f32>

  %buf1 = memref.alloca() : memref<f32>
  memref.store %init1, %buf1[] : memref<f32>
  scf.for %j = %lb to %ub step %step {
    %old = memref.load %buf1[] : memref<f32>
    %new = arith.mulf %old, %old : f32
    memref.store %new, %buf1[] : memref<f32>
  }
  %final1 = memref.load %buf1[] : memref<f32>

  return %final0, %final1 : f32, f32
}

// -----

// Negative: alloca passed to a function call (unknown user). Should not promote.
// CHECK-LABEL: func.func @negative_indirect_use
// CHECK: memref.alloca
func.func @negative_indirect_use(%init: f32, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    %old = memref.load %buf[] : memref<f32>
    %new = arith.addf %old, %old : f32
    memref.store %new, %buf[] : memref<f32>
  }
  "test.use_memref"(%buf) : (memref<f32>) -> ()
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}

// -----

// Negative: load/store inside scf.if within the loop. Promoting would change
// semantics since the load/store may not execute every iteration.
// CHECK-LABEL: func.func @negative_conditional_in_loop
// CHECK: memref.alloca
func.func @negative_conditional_in_loop(%init: f32, %cond: i1, %lb: index, %ub: index, %step: index) -> f32 {
  %buf = memref.alloca() : memref<f32>
  memref.store %init, %buf[] : memref<f32>
  scf.for %i = %lb to %ub step %step {
    scf.if %cond {
      %old = memref.load %buf[] : memref<f32>
      %new = arith.addf %old, %old : f32
      memref.store %new, %buf[] : memref<f32>
    }
  }
  %final = memref.load %buf[] : memref<f32>
  return %final : f32
}
