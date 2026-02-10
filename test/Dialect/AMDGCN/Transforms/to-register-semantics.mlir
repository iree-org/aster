// RUN: aster-opt --amdgcn-to-register-semantics --split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @range_amdgcn.allocations() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_6]], %[[ALLOCA_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[ALLOCA_7]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_6]], %[[ALLOCA_7]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_6]], %[[ALLOCA_7]], %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[MAKE_REGISTER_RANGE_0]] ins %[[MAKE_REGISTER_RANGE_2]] : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr_range<[? : ? + 2]>) -> !amdgcn.vgpr_range<[? : ? + 2]>
// CHECK:           amdgcn.test_inst ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr_range<[? : ? + 4]>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @range_amdgcn.allocations() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %4 = amdgcn.alloca : !amdgcn.vgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.vgpr
  %7 = lsir.copy %6, %0 : !amdgcn.vgpr, !amdgcn.vgpr
  %8 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %9 = amdgcn.alloca : !amdgcn.vgpr
  %10 = lsir.copy %9, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %11 = amdgcn.test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  %12 = amdgcn.make_register_range %7, %10 : !amdgcn.vgpr, !amdgcn.vgpr
  %13 = amdgcn.make_register_range %7, %10, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  %14 = amdgcn.make_register_range %4, %5 : !amdgcn.vgpr, !amdgcn.vgpr
  %15 = amdgcn.test_inst outs %12 ins %14 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 2]>
  amdgcn.test_inst ins %13 : (!amdgcn.vgpr_range<[? + 4]>) -> ()
  amdgcn.test_inst ins %8, %11 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @no_interferencemixed_with_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_2]] : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_3]] : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           return
// CHECK:         }
func.func @no_interferencemixed_with_values() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.sgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.test_inst outs %2 : (!amdgcn.sgpr) -> !amdgcn.sgpr
  %5 = amdgcn.test_inst outs %3 : (!amdgcn.sgpr) -> !amdgcn.sgpr
  %6 = amdgcn.test_inst outs %0 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %7 = amdgcn.test_inst outs %1 ins %5 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  func.return
}

// -----
// CHECK-LABEL:   func.func @interference_mixed_with_reuse_with_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_6]], %[[ALLOCA_3]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_6]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @interference_mixed_with_reuse_with_values() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.sgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.vgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.sgpr
  %7 = lsir.copy %6, %3 : !amdgcn.sgpr, !amdgcn.sgpr
  %8:2 = amdgcn.test_inst outs %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %9:2 = amdgcn.test_inst outs %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  %10 = amdgcn.test_inst outs %0 ins %8#0, %8#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %11 = amdgcn.test_inst outs %1 ins %7, %9#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %10, %11 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @interference_mixed_all_live_undef_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]], %[[ALLOCA_4]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @interference_mixed_all_live_undef_values() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.sgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.vgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  %7 = amdgcn.test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %6, %7, %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @interference_mixed_all_live_with_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]], %[[ALLOCA_4]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @interference_mixed_all_live_with_values() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.sgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.vgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6:2 = amdgcn.test_inst outs %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %7:2 = amdgcn.test_inst outs %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  %8 = amdgcn.test_inst outs %0 ins %6#0, %7#0 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  %9 = amdgcn.test_inst outs %1 ins %6#1, %7#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %8, %9, %7#0, %7#1 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @no_interference_cf_with_values() {
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ALLOCA_6]], %[[ALLOCA_0]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]], %[[ALLOCA_4]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[ALLOCA_7]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[TEST_INST_4:.*]] = amdgcn.test_inst outs %[[ALLOCA_7]] ins %[[ALLOCA_0]], %[[ALLOCA_4]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[TEST_INST_5:.*]] = amdgcn.test_inst outs %[[ALLOCA_6]] ins %[[ALLOCA_1]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @no_interference_cf_with_values() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.sgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.vgpr
  %7:2 = amdgcn.test_inst outs %3, %4 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %8:2 = amdgcn.test_inst outs %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  %9 = amdgcn.alloca : !amdgcn.vgpr
  %10 = lsir.copy %9, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %11 = amdgcn.test_inst outs %1 ins %7#0, %8#0 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  %12 = amdgcn.alloca : !amdgcn.vgpr
  %13 = lsir.copy %12, %2 : !amdgcn.vgpr, !amdgcn.vgpr
  %14 = amdgcn.test_inst outs %2 ins %7#1, %8#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %15 = amdgcn.test_inst outs %13 ins %11, %8#0 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %16 = amdgcn.test_inst outs %10 ins %14, %8#1 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  func.return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @interference_cf_with_values() {
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_4]] ins %[[ALLOCA_0]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[ALLOCA_5]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @interference_cf_with_values() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.sgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.vgpr
  %7 = amdgcn.test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  %8 = amdgcn.test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %9 = amdgcn.test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %10 = amdgcn.test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  amdgcn.test_inst ins %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @existing_regs_with_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr<5>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_8:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_9:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_10:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_11:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_12:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_5]], %[[ALLOCA_6]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
// CHECK:           %[[TEST_INST_2:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_7]], %[[ALLOCA_8]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_3:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_9]], %[[ALLOCA_10]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_4:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_11]], %[[ALLOCA_12]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           %[[TEST_INST_5:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_2]], %[[ALLOCA_5]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<0>
// CHECK:           %[[TEST_INST_6:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_3]], %[[ALLOCA_6]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<1>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_9]], %[[ALLOCA_10]], %[[ALLOCA_11]], %[[ALLOCA_12]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_7]], %[[ALLOCA_8]], %[[ALLOCA_4]], %[[ALLOCA_5]], %[[ALLOCA_6]], %[[ALLOCA_2]], %[[ALLOCA_3]], %[[MAKE_REGISTER_RANGE_0]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<5>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[? : ? + 4]>) -> ()
// CHECK:           return
// CHECK:         }
func.func @existing_regs_with_values() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<0>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  %2 = amdgcn.alloca : !amdgcn.sgpr<0>
  %3 = amdgcn.alloca : !amdgcn.sgpr<2>
  %4 = amdgcn.alloca : !amdgcn.sgpr<5>
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.vgpr
  %7 = amdgcn.alloca : !amdgcn.sgpr
  %8 = amdgcn.alloca : !amdgcn.sgpr
  %9 = amdgcn.alloca : !amdgcn.sgpr
  %10 = amdgcn.alloca : !amdgcn.sgpr
  %11 = amdgcn.alloca : !amdgcn.sgpr
  %12 = amdgcn.alloca : !amdgcn.sgpr
  %13:2 = amdgcn.test_inst outs %2, %3 : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
  %14:2 = amdgcn.test_inst outs %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
  %15:2 = amdgcn.test_inst outs %7, %8 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %16:2 = amdgcn.test_inst outs %9, %10 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %17:2 = amdgcn.test_inst outs %11, %12 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  %18 = amdgcn.test_inst outs %0 ins %13#0, %14#0 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr) -> !amdgcn.vgpr<0>
  %19 = amdgcn.test_inst outs %1 ins %13#1, %14#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr) -> !amdgcn.vgpr<1>
  %20 = amdgcn.make_register_range %16#0, %16#1, %17#0, %17#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  amdgcn.test_inst ins %18, %19, %15#0, %15#1, %4, %14#0, %14#1, %13#0, %13#1, %20 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr<5>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[? + 4]>) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @reg_interference_with_values() {
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_1]], %[[ALLOCA_2]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:           amdgcn.reg_interference %[[ALLOCA_0]], %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_2:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_4]], %[[ALLOCA_5]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[TEST_INST_3:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_6]], %[[ALLOCA_7]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_6]], %[[ALLOCA_7]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:           amdgcn.reg_interference %[[ALLOCA_4]], %[[ALLOCA_1]], %[[ALLOCA_3]], %[[ALLOCA_7]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_5]] : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @reg_interference_with_values() {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.sgpr
  %2:2 = amdgcn.test_inst outs %0, %1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.test_inst ins %2#0, %2#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.sgpr
  %5:2 = amdgcn.test_inst outs %3, %4 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.test_inst ins %2#1, %5#0 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
  amdgcn.reg_interference %2#0, %5#0, %5#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  %6 = amdgcn.alloca : !amdgcn.sgpr
  %7 = amdgcn.alloca : !amdgcn.sgpr
  %8:2 = amdgcn.test_inst outs %6, %7 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.test_inst ins %8#0, %8#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
  %9 = amdgcn.alloca : !amdgcn.sgpr
  %10 = amdgcn.alloca : !amdgcn.sgpr
  %11:2 = amdgcn.test_inst outs %9, %10 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.test_inst ins %11#0, %11#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
  amdgcn.reg_interference %8#0, %2#1, %5#1, %11#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  amdgcn.test_inst ins %2#0, %8#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
  func.return
}

// -----
// CHECK-LABEL:   func.func @test_index_bxmxnxk() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ALLOCA_3]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr<?>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_and_b32 outs %[[ALLOCA_4]] ins %[[ALLOCA_3]], %[[CONSTANT_0]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, i32
// CHECK-DAG:       %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_5]], %[[ALLOCA_6]] : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_1:.*]] = amdgcn.load s_load_dwordx2 dest %[[MAKE_REGISTER_RANGE_1]] addr %[[MAKE_REGISTER_RANGE_0]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr_range<[? : ? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_4]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @rand() -> i1
func.func @test_index_bxmxnxk() {
  %c42_i32 = arith.constant 42 : i32
  %0 = amdgcn.alloca : !amdgcn.sgpr<2>
  %1 = amdgcn.alloca : !amdgcn.sgpr<0>
  %2 = amdgcn.alloca : !amdgcn.sgpr<1>
  %3 = amdgcn.make_register_range %1, %2 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  %4 = amdgcn.alloca : !amdgcn.sgpr
  %result, %token = amdgcn.load s_load_dword dest %4 addr %3 offset c(%c42_i32) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
  amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
  %5 = amdgcn.alloca : !amdgcn.sgpr
  %6 = amdgcn.sop2 s_and_b32 outs %5 ins %result, %c42_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
  %7 = amdgcn.alloca : !amdgcn.sgpr
  %8 = amdgcn.alloca : !amdgcn.sgpr
  %9 = amdgcn.make_register_range %7, %8 : !amdgcn.sgpr, !amdgcn.sgpr
  %result_0, %token_1 = amdgcn.load s_load_dwordx2 dest %9 addr %3 offset c(%c42_i32) : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
  amdgcn.test_inst ins %0, %6 : (!amdgcn.sgpr<2>, !amdgcn.sgpr) -> ()
  func.return
}

// -----
func.func private @rand() -> i1
// CHECK-LABEL:   func.func @split_range() {
// CHECK:           %[[VAL_0:.*]] = call @rand() : () -> i1
// CHECK-DAG:       %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK-DAG:       %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK-DAG:       %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_1]] ins %[[ALLOCA_2]] : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[ALLOCA_3]] ins %[[MAKE_REGISTER_RANGE_0]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr_range<[? : ? + 2]>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[ALLOCA_4]] ins %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_3]], %[[ALLOCA_4]], %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
// CHECK:           return
// CHECK:         }
func.func @split_range() {
  %0 = func.call @rand() : () -> i1
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.sgpr
  %4 = amdgcn.alloca : !amdgcn.sgpr
  %5 = amdgcn.alloca : !amdgcn.vgpr
  %6 = amdgcn.alloca : !amdgcn.vgpr
  %7 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr, !amdgcn.vgpr
  %8 = amdgcn.test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %9 = amdgcn.test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %10 = amdgcn.test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %11, %12 = amdgcn.split_register_range %7 : !amdgcn.vgpr_range<[? + 2]>
  amdgcn.test_inst ins %5, %6, %11, %12 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

// -----

// CHECK-LABEL:   func.func @dealloc() -> !amdgcn.vgpr {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ALLOCA_0]] : !amdgcn.vgpr<?> to !amdgcn.vgpr {__to_register_semantics__}
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @dealloc() -> !amdgcn.vgpr {
  %a = amdgcn.alloca : !amdgcn.vgpr<?>
  %r = amdgcn.dealloc_cast %a : !amdgcn.vgpr<?>
  return %r : !amdgcn.vgpr
}
