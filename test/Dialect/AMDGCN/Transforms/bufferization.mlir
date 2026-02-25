// RUN: aster-opt %s --aster-amdgcn-bufferization --split-input-file | FileCheck %s

// Simple diamond CFG: two allocas merge at block argument.
// The pass should insert copies before each branch.
// CHECK-LABEL: kernel @bufferization_phi_copies_1 {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             %[[VAL_4:.*]] = test_inst outs %[[COPY_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @bufferization_phi_copies_1 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_phi_copies_1 {
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
}

// -----

// CHECK-LABEL: kernel @bufferization_same_phi_value {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @bufferization_same_phi_value target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_same_phi_value {
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
}

// -----

// Test SGPR type: should insert copies.
// CHECK-LABEL: kernel @bufferization_sgpr_copies {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_2]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_3]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr<?>
// CHECK:             test_inst ins %[[COPY_0]] : (!amdgcn.sgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }

amdgcn.module @bufferization_sgpr_copies target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_sgpr_copies {
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
}

// -----

// Values derived from allocas (not raw allocas) - should still insert copies.

// CHECK-LABEL: kernel @bufferization_derived_values {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[VAL_5:.*]] = test_inst outs %[[VAL_2]] ins %[[VAL_4]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_6:.*]] = test_inst outs %[[VAL_3]] ins %[[VAL_4]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             lsir.copy %[[VAL_0]], %[[VAL_6]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             test_inst ins %[[COPY_0]] : (!amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @bufferization_derived_values target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_derived_values {
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
}

// -----

// Same alloca written twice in ^bb0; the first value (%v1) is used in a
// successor block. The clobber copy must replace that cross-block use.
// CHECK-LABEL: kernel @cross_block_clobber {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = test_inst outs %[[VAL_3]] ins %[[VAL_1]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             test_inst ins %[[VAL_2]] : (!amdgcn.vgpr) -> ()
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             test_inst ins %[[VAL_4]] : (!amdgcn.vgpr) -> ()
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @cross_block_clobber target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @cross_block_clobber {
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
}

// -----
// CHECK-LABEL: kernel @too_few_allocas {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_1:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = test_inst outs %[[VAL_2]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_5:.*]] = test_inst outs %[[VAL_4]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             test_inst ins %[[VAL_1]], %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @too_few_allocas target = <gfx942> isa = <cdna3> {
  kernel @too_few_allocas {
    %0 = alloca : !amdgcn.vgpr
    %1 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %1, %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @bufferization_loop_backedge {
// CHECK:             %[[COMMON:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[ARG_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[INIT_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[SGPR:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[INIT:.*]] = test_inst outs %[[INIT_ALLOC]] ins %[[SGPR]]
// CHECK:             lsir.copy %[[COMMON]], %[[INIT]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[ACC:.*]] = lsir.copy %[[ARG_ALLOC]], %[[COMMON]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             %[[COND:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[NEXT_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[NEXT:.*]] = test_inst outs %[[NEXT_ALLOC]] ins %[[ACC]]
// CHECK:             lsir.copy %[[COMMON]], %[[NEXT]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK:           ^bb2:
// CHECK:             test_inst ins %[[ACC]] : (!amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @loop_backedge_test target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_loop_backedge {
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
}

// -----

// Note: BBArgs processed in order; later BBArg allocas inserted at entry start,
// so %y's allocas appear before %x's in the output.
// CHECK-LABEL: kernel @bufferization_swap {
// CHECK:             %[[COMMON_Y:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[ARG_Y:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COMMON_X:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[ARG_X:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[A_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[B_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[SGPR:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[A:.*]] = test_inst outs %[[A_ALLOC]] ins %[[SGPR]]
// CHECK:             %[[B:.*]] = test_inst outs %[[B_ALLOC]] ins %[[SGPR]]
// CHECK:             lsir.copy %[[COMMON_X]], %[[A]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             lsir.copy %[[COMMON_Y]], %[[B]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[X:.*]] = lsir.copy %[[ARG_Y]], %[[COMMON_Y]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             %[[Y:.*]] = lsir.copy %[[ARG_X]], %[[COMMON_X]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             %[[COND:.*]] = func.call @rand() : () -> i1
// CHECK:             lsir.copy %[[COMMON_X]], %[[X]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             lsir.copy %[[COMMON_Y]], %[[Y]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK:           ^bb2:
// CHECK:             test_inst ins %[[Y]], %[[X]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @swap_test target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_swap {
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
}

// -----

// Note: BBArgs processed in order; later BBArg allocas inserted at entry start,
// so %y's allocas appear before %x's in the output.
// CHECK-LABEL: kernel @bufferization_multi_bbarg {
// CHECK:             %[[COMMON_Y:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[ARG_Y:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COMMON_X:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             %[[ARG_X:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[A_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[B_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[C_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[D_ALLOC:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[SGPR:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[A:.*]] = test_inst outs %[[A_ALLOC]] ins %[[SGPR]]
// CHECK:             %[[B:.*]] = test_inst outs %[[B_ALLOC]] ins %[[SGPR]]
// CHECK:             %[[C:.*]] = test_inst outs %[[C_ALLOC]] ins %[[SGPR]]
// CHECK:             %[[D:.*]] = test_inst outs %[[D_ALLOC]] ins %[[SGPR]]
// CHECK:             cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             lsir.copy %[[COMMON_X]], %[[A]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             lsir.copy %[[COMMON_Y]], %[[B]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             lsir.copy %[[COMMON_X]], %[[C]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             lsir.copy %[[COMMON_Y]], %[[D]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[RY:.*]] = lsir.copy %[[ARG_Y]], %[[COMMON_Y]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             %[[RX:.*]] = lsir.copy %[[ARG_X]], %[[COMMON_X]] : !amdgcn.vgpr, !amdgcn.vgpr<?>
// CHECK:             test_inst ins %[[RX]], %[[RY]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @multi_bbarg_test target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @bufferization_multi_bbarg {
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
}
