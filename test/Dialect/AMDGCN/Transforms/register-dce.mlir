// RUN: aster-opt %s --amdgcn-register-dce | FileCheck %s

// CHECK-LABEL:   func.func private @rand() -> i1

func.func private @rand() -> i1
// CHECK-LABEL:   func.func @cross_block_clobber() {
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_2]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
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
  %4 = lsir.copy %3, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = amdgcn.test_inst outs %1 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %6 = amdgcn.test_inst outs %3 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
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
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           return
// CHECK:         }
func.func @dead_copy_0() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %3 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
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
  %2 = lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL:   func.func @live_copy() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @live_copy() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
  return
}
