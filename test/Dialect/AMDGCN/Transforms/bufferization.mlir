// RUN: aster-opt %s --aster-amdgcn-bufferization --split-input-file | FileCheck %s

// Simple diamond CFG: two allocas merge at block argument.
// The pass should insert copies before each branch.
// CHECK-LABEL:   amdgcn.module @bufferization_phi_copies_1 target = <gfx942> isa = <cdna3> {
// CHECK:           func.func private @rand() -> i1
// CHECK:           kernel @bufferization_phi_copies_1 {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK-DAG:         %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK-DAG:         %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_3]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_2]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_2:.*]] = lsir.copy %[[VAL_4]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_3:.*]] = lsir.copy %[[VAL_2]], %[[COPY_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_5:.*]] = dealloc_cast %[[VAL_2]] : !amdgcn.vgpr<?>
// CHECK:             %[[VAL_6:.*]] = test_inst outs %[[VAL_5]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
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

// CHECK-LABEL:   amdgcn.module @bufferization_same_phi_value target = <gfx942> isa = <cdna3> {
// CHECK:           func.func private @rand() -> i1
// CHECK:           kernel @bufferization_same_phi_value {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK-DAG:         %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_2]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_1]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_2:.*]] = lsir.copy %[[VAL_3]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_3:.*]] = lsir.copy %[[VAL_1]], %[[COPY_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_4:.*]] = dealloc_cast %[[VAL_1]] : !amdgcn.vgpr<?>
// CHECK:             test_inst ins %[[VAL_4]] : (!amdgcn.vgpr) -> ()
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
// CHECK-LABEL:   amdgcn.module @bufferization_sgpr_copies target = <gfx942> isa = <cdna3> {
// CHECK:           func.func private @rand() -> i1
// CHECK:           kernel @bufferization_sgpr_copies {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[VAL_0:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:         %[[VAL_1:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:         %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_3]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_2]], %[[COPY_0]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[COPY_2:.*]] = lsir.copy %[[VAL_4]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:             %[[COPY_3:.*]] = lsir.copy %[[VAL_2]], %[[COPY_2]] : !amdgcn.sgpr<?>, !amdgcn.sgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_5:.*]] = dealloc_cast %[[VAL_2]] : !amdgcn.sgpr<?>
// CHECK:             test_inst ins %[[VAL_5]] : (!amdgcn.sgpr) -> ()
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
// CHECK-LABEL:   amdgcn.module @bufferization_derived_values target = <gfx942> isa = <cdna3> {
// CHECK:           func.func private @rand() -> i1
// CHECK:           kernel @bufferization_derived_values {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK-DAG:         %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK-DAG:         %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK:             %[[VAL_3:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_2]] : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_6:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_6]], %[[VAL_3]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_5]], %[[COPY_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_7:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_2:.*]] = lsir.copy %[[VAL_7]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[COPY_3:.*]] = lsir.copy %[[VAL_5]], %[[COPY_2]] : !amdgcn.vgpr<?>, !amdgcn.vgpr
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             %[[VAL_8:.*]] = dealloc_cast %[[VAL_5]] : !amdgcn.vgpr<?>
// CHECK:             test_inst ins %[[VAL_8]] : (!amdgcn.vgpr) -> ()
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

// CHECK-LABEL:   amdgcn.module @too_few_allocas target = <gfx942> isa = <cdna3> {
// CHECK:           kernel @too_few_allocas {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = test_inst outs %[[COPY_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_4]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_5:.*]] = test_inst outs %[[VAL_2]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             test_inst ins %[[COPY_1]], %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
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
