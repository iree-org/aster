// RUN: aster-opt %s --aster-amdgcn-bufferization --split-input-file | FileCheck %s

// Simple diamond CFG: two allocas merge at block argument.
// The pass should insert copies before each branch.

func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_phi_copies_1 {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[COPY_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_phi_copies_1 {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr
  %2 = alloca : !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3(%1 : !amdgcn.vgpr)
^bb2:  // pred: ^bb0
  cf.br ^bb3(%2 : !amdgcn.vgpr)
^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
  %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  end_kernel
}

// -----

func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_same_phi_value {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_same_phi_value {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%1 : !amdgcn.vgpr)
^bb2:
  cf.br ^bb3(%1 : !amdgcn.vgpr)
^bb3(%3: !amdgcn.vgpr):
  test_inst ins %3 : (!amdgcn.vgpr) -> ()
  end_kernel
}

// -----

// Test SGPR type: should insert copies.

func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_sgpr_copies {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_3]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr<?>
// CHECK:           test_inst ins %[[COPY_0]] : (!amdgcn.sgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_sgpr_copies {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.sgpr
  %2 = alloca : !amdgcn.sgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%1 : !amdgcn.sgpr)
^bb2:
  cf.br ^bb3(%2 : !amdgcn.sgpr)
^bb3(%3: !amdgcn.sgpr):
  test_inst ins %3 : (!amdgcn.sgpr) -> ()
  end_kernel
}

// -----

// Values derived from allocas (not raw allocas) - should still insert copies.
func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_derived_values {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_2]] ins %[[VAL_4]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_3]] ins %[[VAL_4]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_6]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_derived_values {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr
  %2 = alloca : !amdgcn.vgpr
  %3 = alloca : !amdgcn.sgpr
  %v1 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %v2 = test_inst outs %2 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%v1 : !amdgcn.vgpr)
^bb2:
  cf.br ^bb3(%v2 : !amdgcn.vgpr)
^bb3(%val: !amdgcn.vgpr):
  test_inst ins %val : (!amdgcn.vgpr) -> ()
  end_kernel
}

// -----

// Same alloca written twice in ^bb0; the first value (%v1) is used in a
// successor block. The clobber copy must replace that cross-block use.
func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @cross_block_clobber {
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_3]] ins %[[VAL_1]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           test_inst ins %[[VAL_2]] : (!amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           test_inst ins %[[VAL_4]] : (!amdgcn.vgpr) -> ()
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @cross_block_clobber {
  %cond = func.call @rand() : () -> i1
  %0 = alloca : !amdgcn.vgpr
  %1 = alloca : !amdgcn.sgpr
  %v1 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %v2 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  test_inst ins %v1 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb2:
  test_inst ins %v2 : (!amdgcn.vgpr) -> ()
  cf.br ^bb3
^bb3:
  end_kernel
}

// -----

// CHECK-LABEL:   amdgcn.kernel @too_few_allocas {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = test_inst outs %[[VAL_2]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_4]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           test_inst ins %[[VAL_1]], %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @too_few_allocas {
  %0 = alloca : !amdgcn.vgpr
  %1 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  test_inst ins %1, %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
  end_kernel
}

// -----

func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_loop_backedge {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_2]] ins %[[VAL_3]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_4]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_5]] ins %[[COPY_0]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           cf.cond_br %[[CALL_0]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_6]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb4:
// CHECK:           test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_loop_backedge {
  %0 = alloca : !amdgcn.vgpr
  %1 = alloca : !amdgcn.sgpr
  %init = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.br ^header(%init : !amdgcn.vgpr)
^header(%acc: !amdgcn.vgpr):
  %cond = func.call @rand() : () -> i1
  %2 = alloca : !amdgcn.vgpr
  %next = test_inst outs %2 ins %acc : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.cond_br %cond, ^header(%next : !amdgcn.vgpr), ^exit
^exit:
  test_inst ins %acc : (!amdgcn.vgpr) -> ()
  end_kernel
}

// -----

// Note: BBArgs processed in order; later BBArg allocas inserted at entry start,
// so %y's allocas appear before %x's in the output.
func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_swap {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_4]] ins %[[VAL_6]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_8:.*]] = test_inst outs %[[VAL_5]] ins %[[VAL_6]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           lsir.copy %[[VAL_2]], %[[VAL_7]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_8]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_3]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[CALL_0]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           lsir.copy %[[VAL_2]], %[[COPY_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           lsir.copy %[[VAL_0]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb4:
// CHECK:           test_inst ins %[[COPY_0]], %[[COPY_1]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
amdgcn.kernel @bufferization_swap {
  %0 = alloca : !amdgcn.vgpr
  %1 = alloca : !amdgcn.vgpr
  %2 = alloca : !amdgcn.sgpr
  %a = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %b = test_inst outs %1 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.br ^loop(%a, %b : !amdgcn.vgpr, !amdgcn.vgpr)
^loop(%x: !amdgcn.vgpr, %y: !amdgcn.vgpr):
  %cond = func.call @rand() : () -> i1
  cf.cond_br %cond, ^loop(%y, %x : !amdgcn.vgpr, !amdgcn.vgpr), ^exit
^exit:
  test_inst ins %x, %y : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  end_kernel
}

// -----

// Note: BBArgs processed in order; later BBArg allocas inserted at entry start,
// so %y's allocas appear before %x's in the output.
func.func private @rand() -> i1
// CHECK-LABEL:   amdgcn.kernel @bufferization_multi_bbarg {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_5:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_7:.*]] = alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_8:.*]] = alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_9:.*]] = test_inst outs %[[VAL_4]] ins %[[VAL_8]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_10:.*]] = test_inst outs %[[VAL_5]] ins %[[VAL_8]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_11:.*]] = test_inst outs %[[VAL_6]] ins %[[VAL_8]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_12:.*]] = test_inst outs %[[VAL_7]] ins %[[VAL_8]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb3
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           lsir.copy %[[VAL_2]], %[[VAL_9]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_10]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb3:
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           lsir.copy %[[VAL_2]], %[[VAL_11]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           lsir.copy %[[VAL_0]], %[[VAL_12]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[VAL_3]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           test_inst ins %[[COPY_0]], %[[COPY_1]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @bufferization_multi_bbarg {
  %0 = alloca : !amdgcn.vgpr
  %1 = alloca : !amdgcn.vgpr
  %2 = alloca : !amdgcn.vgpr
  %3 = alloca : !amdgcn.vgpr
  %4 = alloca : !amdgcn.sgpr
  %a = test_inst outs %0 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %b = test_inst outs %1 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %c = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %d = test_inst outs %3 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %cond = func.call @rand() : () -> i1
  cf.cond_br %cond, ^left, ^right
^left:
  cf.br ^merge(%a, %b : !amdgcn.vgpr, !amdgcn.vgpr)
^right:
  cf.br ^merge(%c, %d : !amdgcn.vgpr, !amdgcn.vgpr)
^merge(%x: !amdgcn.vgpr, %y: !amdgcn.vgpr):
  test_inst ins %x, %y : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  end_kernel
}

// -----

func.func private @rand() -> i1

// CHECK-LABEL:   func.func @test_copy_loc() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_4]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_5]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           lsir.copy %[[ALLOCA_2]], %[[TEST_INST_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           lsir.copy %[[ALLOCA_0]], %[[TEST_INST_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_1]], %[[ALLOCA_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[ALLOCA_3]], %[[ALLOCA_2]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[COPY_1]] : (!amdgcn.vgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_copy_loc() {
  // This test checks that copies are inserted right before their first use.
  // This corresponds to the insertion point used in the `handleBlockArgument` function.
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb1(%2, %3 : !amdgcn.vgpr, !amdgcn.vgpr)
^bb1(%4: !amdgcn.vgpr, %5: !amdgcn.vgpr):  // pred: ^bb0
  amdgcn.test_inst ins %5 : (!amdgcn.vgpr) -> ()
  amdgcn.test_inst ins %4 : (!amdgcn.vgpr) -> ()
  return
}

// CHECK-LABEL:   func.func @test_copy_edge_liveness() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_2]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           lsir.copy %[[ALLOCA_0]], %[[TEST_INST_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_1]], %[[ALLOCA_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           lsir.copy %[[ALLOCA_0]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:           cf.br ^bb2
// CHECK:         ^bb4:
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_3]] ins %[[COPY_0]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           amdgcn.test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_copy_edge_liveness() {
  // This test checks that in the exit block we use the copy target instead of
  // the original source. This corrresponds to the `optimizeLiveRanges` function.
  // Further, it tests that the target allocation is not clobbered nor used as
  // an output operand.
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb1(%2 : !amdgcn.vgpr)
^bb1(%3: !amdgcn.vgpr):  // 2 preds: ^bb0, ^bb1
  %4 = call @rand() : () -> i1
  cf.cond_br %4, ^bb1(%3 : !amdgcn.vgpr), ^bb2
^bb2:  // pred: ^bb1
  %5 = amdgcn.test_inst outs %3 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr)
  amdgcn.test_inst ins %3 : (!amdgcn.vgpr) -> ()
  return
}
