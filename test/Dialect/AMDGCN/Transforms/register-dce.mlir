// RUN: aster-opt %s --amdgcn-register-dce | FileCheck %s

// CHECK-LABEL:   func.func private @rand() -> i1

func.func private @rand() -> i1
// CHECK-LABEL:   func.func @cross_block_clobber() {
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_2]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
func.func @cross_block_clobber() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.sgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %3, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst outs %1 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
  amdgcn.test_inst outs %3 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
  cf.br ^bb3
^bb2:  // pred: ^bb0
  amdgcn.test_inst ins %3 : (!amdgcn.vgpr<?>) -> ()
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  return
}

// CHECK-LABEL:   func.func @dead_copy_0() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst outs %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @dead_copy_0() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL:   func.func @dead_copy_1() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @dead_copy_1() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL:   func.func @live_copy() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           lsir.copy %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @live_copy() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
  return
}

// Verify that a range copy is preserved when the target allocas are used
// downstream. This was a bug where DCE checked the make_register_range
// result value for liveness instead of the underlying allocas.
// CHECK-LABEL:   func.func @live_range_copy() {
// CHECK:           %[[A0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[A1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[S0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[S1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst outs %[[S0]]
// CHECK:           amdgcn.test_inst outs %[[S1]]
// CHECK:           %[[TR:.*]] = amdgcn.make_register_range %[[A0]], %[[A1]]
// CHECK:           %[[SR:.*]] = amdgcn.make_register_range %[[S0]], %[[S1]]
// CHECK:           lsir.copy %[[TR]], %[[SR]]
// CHECK:           amdgcn.test_inst ins %[[A0]], %[[A1]]
// CHECK:           return
// CHECK:         }
func.func @live_range_copy() {
  %a0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %a1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %s0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %s1 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %s0 : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst outs %s1 : (!amdgcn.vgpr<?>) -> ()
  %target = amdgcn.make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %source = amdgcn.make_register_range %s0, %s1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %target, %source : !amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr_range<[? : ? + 2]>
  amdgcn.test_inst ins %a0, %a1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// Verify that a dead range copy is still eliminated.
// CHECK-LABEL:   func.func @dead_range_copy() {
// CHECK:           %[[A0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[A1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[S0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[S1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-NOT:       lsir.copy
// CHECK:           return
// CHECK:         }
func.func @dead_range_copy() {
  %a0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %a1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %s0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %s1 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst outs %s0 : (!amdgcn.vgpr<?>) -> ()
  amdgcn.test_inst outs %s1 : (!amdgcn.vgpr<?>) -> ()
  %target = amdgcn.make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %source = amdgcn.make_register_range %s0, %s1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %target, %source : !amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr_range<[? : ? + 2]>
  return
}
