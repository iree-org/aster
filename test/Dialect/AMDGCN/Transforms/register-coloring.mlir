// RUN: aster-opt %s --amdgcn-register-coloring --cse --split-input-file | FileCheck %s

// CHECK-LABEL:   amdgcn.kernel @range_allocations {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<6>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<7>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK-DAG:       %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<4>
// CHECK:           %[[COPY_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_6]], %[[VAL_0]] : (!amdgcn.vgpr<4>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<4>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_8:.*]] = alloca : !amdgcn.vgpr<5>
// CHECK:           %[[COPY_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[VAL_8]], %[[VAL_1]] : (!amdgcn.vgpr<5>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<5>
// CHECK:           %[[VAL_9:.*]] = test_inst outs %[[VAL_1]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_10:.*]] = make_register_range %[[VAL_6]], %[[VAL_8]] : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
// CHECK:           %[[VAL_11:.*]] = make_register_range %[[VAL_6]], %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
// CHECK:           %[[VAL_12:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:           %[[VAL_13:.*]] = test_inst outs %[[VAL_10]] ins %[[VAL_12]] : (!amdgcn.vgpr_range<[4 : 6]>, !amdgcn.vgpr_range<[2 : 4]>) -> !amdgcn.vgpr_range<[4 : 6]>
// CHECK:           test_inst ins %[[VAL_11]] : (!amdgcn.vgpr_range<[4 : 8]>) -> ()
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @range_allocations {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = lsir.copy %6, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %8 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %9 = alloca : !amdgcn.vgpr<?>
  %10 = lsir.copy %9, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %11 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %12 = make_register_range %6, %9 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %13 = make_register_range %6, %9, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %14 = make_register_range %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %15 = test_inst outs %12 ins %14 : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr_range<[? : ? + 2]>) -> !amdgcn.vgpr_range<[? : ? + 2]>
  test_inst ins %13 : (!amdgcn.vgpr_range<[? : ? + 4]>) -> ()
  test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @no_interference_mixed_undef_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_3:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @no_interference_mixed_undef_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @no_interferencemixed_with_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_3:.*]] = test_inst outs %[[VAL_1]] : (!amdgcn.sgpr<0>) -> !amdgcn.sgpr<0>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_2]] : (!amdgcn.sgpr<1>) -> !amdgcn.sgpr<1>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @no_interferencemixed_with_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = test_inst outs %2 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
  %5 = test_inst outs %3 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
  %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @interference_mixed_with_reuse_undef_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]], %[[VAL_1]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @interference_mixed_with_reuse_undef_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @interference_mixed_with_reuse_with_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:           %[[COPY_0:.*]] = sop1 s_mov_b32 outs %[[VAL_4]] ins %[[VAL_3]] : !amdgcn.sgpr<2>, !amdgcn.sgpr<1>
// CHECK:           %[[VAL_5:.*]]:2 = test_inst outs %[[VAL_2]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:           %[[VAL_6:.*]]:2 = test_inst outs %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_3]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_8:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_4]], %[[VAL_1]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @interference_mixed_with_reuse_with_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.sgpr<?>
  %7 = lsir.copy %6, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %8:2 = test_inst outs %2, %3 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %9:2 = test_inst outs %4, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  %10 = test_inst outs %0 ins %2, %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %11 = test_inst outs %1 ins %6, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @interference_mixed_all_live_undef_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_4]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @interference_mixed_all_live_undef_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1, %4, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @interference_mixed_all_live_with_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_6:.*]]:2 = test_inst outs %[[VAL_2]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:           %[[VAL_7:.*]]:2 = test_inst outs %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:           %[[VAL_8:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_4]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_9:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @interference_mixed_all_live_with_values {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6:2 = test_inst outs %2, %3 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %7:2 = test_inst outs %4, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  %8 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %9 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %0, %1, %4, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @no_interference_cf_undef_values {
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_0]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           end_kernel
// CHECK:         }
func.func private @rand() -> i1
amdgcn.kernel @no_interference_cf_undef_values {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %9 = test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %10 = test_inst outs %6 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @no_interference_cf_with_values {
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_6:.*]]:2 = test_inst outs %[[VAL_2]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:           %[[VAL_7:.*]]:2 = test_inst outs %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:           %[[VAL_8:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_4]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_9:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_10:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_0]], %[[VAL_4]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_11:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_1]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<0>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           end_kernel
// CHECK:         }
func.func private @rand() -> i1
amdgcn.kernel @no_interference_cf_with_values {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7:2 = test_inst outs %3, %4 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %8:2 = test_inst outs %5, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  %9 = alloca : !amdgcn.vgpr<?>
  %10 = lsir.copy %9, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %11 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %12 = alloca : !amdgcn.vgpr<?>
  %13 = lsir.copy %12, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %14 = test_inst outs %2 ins %4, %6 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %15 = test_inst outs %12 ins %1, %5 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %16 = test_inst outs %9 ins %2, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @interference_cf_with_values {
// CHECK:           %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_8:.*]] = test_inst outs %[[VAL_4]] ins %[[VAL_0]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<2>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_9:.*]] = test_inst outs %[[VAL_5]] ins %[[VAL_1]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<3>
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           test_inst ins %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
func.func private @rand() -> i1
amdgcn.kernel @interference_cf_with_values {
  %0 = func.call @rand() : () -> i1
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
  cf.cond_br %0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %9 = test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %10 = test_inst outs %6 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  test_inst ins %5, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @existing_regs_undef_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK-DAG:       %[[VAL_7:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_8:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:       %[[VAL_9:.*]] = alloca : !amdgcn.sgpr<8>
// CHECK-DAG:       %[[VAL_10:.*]] = alloca : !amdgcn.sgpr<9>
// CHECK-DAG:       %[[VAL_11:.*]] = alloca : !amdgcn.sgpr<10>
// CHECK-DAG:       %[[VAL_12:.*]] = alloca : !amdgcn.sgpr<11>
// CHECK:           %[[VAL_13:.*]] = make_register_range %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : !amdgcn.sgpr<8>, !amdgcn.sgpr<9>, !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_13]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<5>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.sgpr<1>, !amdgcn.sgpr<3>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @existing_regs_undef_values {
  %0 = alloca : !amdgcn.vgpr<0>
  %1 = alloca : !amdgcn.vgpr<1>
  %2 = alloca : !amdgcn.sgpr<0>
  %3 = alloca : !amdgcn.sgpr<2>
  %4 = alloca : !amdgcn.sgpr<5>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = alloca : !amdgcn.sgpr<?>
  %8 = alloca : !amdgcn.sgpr<?>
  %9 = alloca : !amdgcn.sgpr<?>
  %10 = alloca : !amdgcn.sgpr<?>
  %11 = alloca : !amdgcn.sgpr<?>
  %12 = alloca : !amdgcn.sgpr<?>
  %13 = make_register_range %9, %10, %11, %12 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  test_inst ins %0, %1, %2, %3, %4, %5, %6, %7, %8, %13 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<5>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr_range<[? : ? + 4]>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @existing_regs_with_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK-DAG:       %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK-DAG:       %[[VAL_7:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[VAL_8:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:       %[[VAL_9:.*]] = alloca : !amdgcn.sgpr<8>
// CHECK-DAG:       %[[VAL_10:.*]] = alloca : !amdgcn.sgpr<9>
// CHECK-DAG:       %[[VAL_11:.*]] = alloca : !amdgcn.sgpr<10>
// CHECK-DAG:       %[[VAL_12:.*]] = alloca : !amdgcn.sgpr<11>
// CHECK:           %[[VAL_13:.*]]:2 = test_inst outs %[[VAL_2]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
// CHECK:           %[[VAL_14:.*]]:2 = test_inst outs %[[VAL_5]], %[[VAL_6]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:           %[[VAL_15:.*]]:2 = test_inst outs %[[VAL_7]], %[[VAL_8]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<3>) -> (!amdgcn.sgpr<1>, !amdgcn.sgpr<3>)
// CHECK:           %[[VAL_16:.*]]:2 = test_inst outs %[[VAL_9]], %[[VAL_10]] : (!amdgcn.sgpr<8>, !amdgcn.sgpr<9>) -> (!amdgcn.sgpr<8>, !amdgcn.sgpr<9>)
// CHECK:           %[[VAL_17:.*]]:2 = test_inst outs %[[VAL_11]], %[[VAL_12]] : (!amdgcn.sgpr<10>, !amdgcn.sgpr<11>) -> (!amdgcn.sgpr<10>, !amdgcn.sgpr<11>)
// CHECK:           %[[VAL_18:.*]] = test_inst outs %[[VAL_0]] ins %[[VAL_2]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_19:.*]] = test_inst outs %[[VAL_1]] ins %[[VAL_3]], %[[VAL_6]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_20:.*]] = make_register_range %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : !amdgcn.sgpr<8>, !amdgcn.sgpr<9>, !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_7]], %[[VAL_8]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_2]], %[[VAL_3]], %[[VAL_20]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.sgpr<3>, !amdgcn.sgpr<5>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @existing_regs_with_values {
  %0 = alloca : !amdgcn.vgpr<0>
  %1 = alloca : !amdgcn.vgpr<1>
  %2 = alloca : !amdgcn.sgpr<0>
  %3 = alloca : !amdgcn.sgpr<2>
  %4 = alloca : !amdgcn.sgpr<5>
  %5 = alloca : !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = alloca : !amdgcn.sgpr<?>
  %8 = alloca : !amdgcn.sgpr<?>
  %9 = alloca : !amdgcn.sgpr<?>
  %10 = alloca : !amdgcn.sgpr<?>
  %11 = alloca : !amdgcn.sgpr<?>
  %12 = alloca : !amdgcn.sgpr<?>
  %13:2 = test_inst outs %2, %3 : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
  %14:2 = test_inst outs %5, %6 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  %15:2 = test_inst outs %7, %8 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %16:2 = test_inst outs %9, %10 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %17:2 = test_inst outs %11, %12 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %18 = test_inst outs %0 ins %2, %5 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<0>
  %19 = test_inst outs %1 ins %3, %6 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<1>
  %20 = make_register_range %9, %10, %11, %12 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  test_inst ins %0, %1, %7, %8, %4, %5, %6, %2, %3, %20 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<5>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[? : ? + 4]>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @test_make_range_liveness_1 {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_4:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_1]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_6:.*]] = test_inst outs %[[VAL_2]] ins %[[VAL_1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_3]] ins %[[VAL_2]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           %[[VAL_8:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_8]], %[[VAL_2]] : (!amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr<2>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @test_make_range_liveness_1 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = alloca : !amdgcn.vgpr<?>
  %3 = alloca : !amdgcn.vgpr<?>
  %4 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %5 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = test_inst outs %2 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %7 = test_inst outs %3 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8, %2 : (!amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @test_make_range_liveness_2 {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_3:.*]] = test_inst outs %[[VAL_1]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_4]] ins %[[VAL_1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_6]] ins %[[VAL_4]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           test_inst ins %[[VAL_4]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           %[[VAL_8:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_8]] : (!amdgcn.vgpr_range<[0 : 2]>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @test_make_range_liveness_2 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8 : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @test_make_range_liveness_3 {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_3:.*]] = test_inst outs %[[VAL_1]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[VAL_5:.*]] = test_inst outs %[[VAL_4]] ins %[[VAL_1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK:           %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_7:.*]] = test_inst outs %[[VAL_6]] ins %[[VAL_4]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           %[[VAL_8:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[VAL_8]] : (!amdgcn.vgpr_range<[0 : 2]>) -> ()
// CHECK:           test_inst ins %[[VAL_4]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @test_make_range_liveness_3 {
  %0 = alloca : !amdgcn.vgpr<?>
  %1 = alloca : !amdgcn.vgpr<?>
  %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %4 = alloca : !amdgcn.vgpr<?>
  %5 = test_inst outs %4 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %6 = alloca : !amdgcn.vgpr<?>
  %7 = test_inst outs %6 ins %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  test_inst ins %8 : (!amdgcn.vgpr_range<[? : ? + 2]>) -> ()
  test_inst ins %4 : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @reg_interference_undef_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:           test_inst ins %[[VAL_1]], %[[VAL_2]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<2>) -> ()
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }

amdgcn.kernel @reg_interference_undef_values {
  %0 = alloca : !amdgcn.sgpr<?>
  %1 = alloca : !amdgcn.sgpr<?>
  test_inst ins %0, %1 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %2 = alloca : !amdgcn.sgpr<?>
  %3 = alloca : !amdgcn.sgpr<?>
  test_inst ins %2, %3 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %0, %2, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5 = alloca : !amdgcn.sgpr<?>
  test_inst ins %4, %5 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %6 = alloca : !amdgcn.sgpr<?>
  %7 = alloca : !amdgcn.sgpr<?>
  test_inst ins %6, %7 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %4, %1, %3, %7 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  end_kernel
}

// -----
// CHECK-LABEL:   amdgcn.kernel @reg_interference_with_values {
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_2:.*]]:2 = test_inst outs %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK-DAG:       %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           %[[VAL_5:.*]]:2 = test_inst outs %[[VAL_3]], %[[VAL_4]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<3>) -> (!amdgcn.sgpr<2>, !amdgcn.sgpr<3>)
// CHECK:           test_inst ins %[[VAL_1]], %[[VAL_3]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<2>) -> ()
// CHECK:           %[[VAL_6:.*]]:2 = test_inst outs %[[VAL_3]], %[[VAL_1]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<2>, !amdgcn.sgpr<1>)
// CHECK:           test_inst ins %[[VAL_3]], %[[VAL_1]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<1>) -> ()
// CHECK:           %[[VAL_7:.*]] = alloca : !amdgcn.sgpr<4>
// CHECK:           %[[VAL_8:.*]]:2 = test_inst outs %[[VAL_3]], %[[VAL_7]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>) -> (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>)
// CHECK:           test_inst ins %[[VAL_3]], %[[VAL_7]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>) -> ()
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK:           end_kernel
// CHECK:         }


amdgcn.kernel @reg_interference_with_values {
  %0 = alloca : !amdgcn.sgpr<?>
  %1 = alloca : !amdgcn.sgpr<?>
  %2:2 = test_inst outs %0, %1 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  test_inst ins %0, %1 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %3 = alloca : !amdgcn.sgpr<?>
  %4 = alloca : !amdgcn.sgpr<?>
  %5:2 = test_inst outs %3, %4 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  test_inst ins %1, %3 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %0, %3, %4 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %6 = alloca : !amdgcn.sgpr<?>
  %7 = alloca : !amdgcn.sgpr<?>
  %8:2 = test_inst outs %6, %7 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  test_inst ins %6, %7 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %9 = alloca : !amdgcn.sgpr<?>
  %10 = alloca : !amdgcn.sgpr<?>
  %11:2 = test_inst outs %9, %10 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  test_inst ins %9, %10 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  reg_interference %6, %1, %4, %10 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  test_inst ins %0, %7 : (!amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL:   amdgcn.kernel @test_index_bxmxnxk arguments <[#amdgcn.buffer_arg<address_space = generic>, #amdgcn.block_dim_arg<x>]> {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK-DAG:       %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[VAL_3:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:           %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = load s_load_dword dest %[[VAL_4]] addr %[[VAL_3]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr<3>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
// CHECK:           %[[VAL_7:.*]] = sop2 s_and_b32 outs %[[VAL_4]] ins %[[VAL_4]], %[[CONSTANT_0]] : !amdgcn.sgpr<3>, !amdgcn.sgpr<3>, i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = load s_load_dwordx2 dest %[[VAL_3]] addr %[[VAL_3]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr_range<[0 : 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_4]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @test_index_bxmxnxk arguments <[#amdgcn.buffer_arg<address_space = generic>, #amdgcn.block_dim_arg<x>]> {
  %c42_i32 = arith.constant 42 : i32
  %0 = alloca : !amdgcn.sgpr<2>
  %1 = alloca : !amdgcn.sgpr<0>
  %2 = alloca : !amdgcn.sgpr<1>
  %3 = make_register_range %1, %2 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  %4 = alloca : !amdgcn.sgpr<?>
  %result, %token = load s_load_dword dest %4 addr %3 offset c(%c42_i32) : dps(!amdgcn.sgpr<?>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
  amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
  %5 = alloca : !amdgcn.sgpr<?>
  %6 = sop2 s_and_b32 outs %5 ins %4, %c42_i32 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, i32
  %7 = alloca : !amdgcn.sgpr<?>
  %8 = alloca : !amdgcn.sgpr<?>
  %9 = make_register_range %7, %8 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %result_0, %token_1 = load s_load_dwordx2 dest %9 addr %3 offset c(%c42_i32) : dps(!amdgcn.sgpr_range<[? : ? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
  test_inst ins %0, %5 : (!amdgcn.sgpr<2>, !amdgcn.sgpr<?>) -> ()
  end_kernel
}

// -----
// CHECK-LABEL:   func.func @copy_single() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_2]] ins %[[ALLOCA_3]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:           return
// CHECK:         }
func.func @copy_single() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.sgpr<?>
  %3 = amdgcn.alloca : !amdgcn.sgpr<?>
  amdgcn.test_inst ins %0, %1, %2, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %4 = lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = lsir.copy %2, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  return
}

// CHECK-LABEL:   func.func @copy_range_1() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %[[ALLOCA_2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[ALLOCA_3]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           return
// CHECK:         }
func.func @copy_range_1() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.sgpr<?>
  %3 = amdgcn.alloca : !amdgcn.sgpr<?>
  amdgcn.test_inst ins %0, %1, %2, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %4 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %2, %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %6 = lsir.copy %4, %5 : !amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.sgpr_range<[? : ? + 2]>
  return
}

// CHECK-LABEL:   func.func @copy_range_2() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.sgpr<2>
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.sgpr<3>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]], %[[ALLOCA_4]], %[[ALLOCA_5]], %[[ALLOCA_6]], %[[ALLOCA_7]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>) -> ()
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %[[ALLOCA_2]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[ALLOCA_3]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_4]] ins %[[ALLOCA_6]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<2>
// CHECK:           %[[VAL_3:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_5]] ins %[[ALLOCA_7]] : !amdgcn.sgpr<1>, !amdgcn.sgpr<3>
// CHECK:           return
// CHECK:         }
func.func @copy_range_2() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.sgpr<?>
  %5 = amdgcn.alloca : !amdgcn.sgpr<?>
  %6 = amdgcn.alloca : !amdgcn.sgpr<?>
  %7 = amdgcn.alloca : !amdgcn.sgpr<?>
  amdgcn.test_inst ins %0, %1, %2, %3, %4, %5, %6, %7 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>) -> ()
  %8 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %9 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %10 = amdgcn.make_register_range %4, %5 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %11 = amdgcn.make_register_range %6, %7 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  %12 = lsir.copy %8, %9 : !amdgcn.vgpr_range<[? : ? + 2]>, !amdgcn.vgpr_range<[? : ? + 2]>
  %13 = lsir.copy %10, %11 : !amdgcn.sgpr_range<[? : ? + 2]>, !amdgcn.sgpr_range<[? : ? + 2]>
  return
}

// CHECK-LABEL:   func.func @copy_elided() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           return
// CHECK:         }
func.func @copy_elided() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  amdgcn.test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = lsir.copy %2, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %0, %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// CHECK-LABEL:   func.func @copies_mixed() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<0>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[TEST_INST_0:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_2]], %[[ALLOCA_0]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_3]], %[[ALLOCA_1]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<3>
// CHECK:           %[[TEST_INST_1:.*]]:2 = amdgcn.test_inst outs %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>)
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]], %[[ALLOCA_3]], %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           amdgcn.test_inst ins %[[ALLOCA_2]], %[[ALLOCA_3]], %[[ALLOCA_0]], %[[ALLOCA_1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           return
// CHECK:         }
func.func @copies_mixed() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2:2 = amdgcn.test_inst outs %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = lsir.copy %3, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = lsir.copy %5, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %7:2 = amdgcn.test_inst outs %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>)
  amdgcn.test_inst ins %3, %5, %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  %8 = amdgcn.alloca : !amdgcn.vgpr<?>
  %9 = lsir.copy %8, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %3, %8, %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
  return
}

// -----
// CHECK-LABEL: @redundant_copies
// CHECK-NOT: v_mov_b32_e32
// CHECK-NOT: s_mov_b32
func.func @redundant_copies() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.sgpr<?>
  %3 = amdgcn.alloca : !amdgcn.sgpr<?>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  amdgcn.sop1 s_mov_b32 outs %2 ins %3 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
  return
}

// -----
// CHECK-LABEL:   amdgcn.kernel @dead_write_interference {
// CHECK-DAG:       %[[LIVE:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[DEAD:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[V0:.*]] = test_inst outs %[[LIVE]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[V1:.*]] = test_inst outs %[[DEAD]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           %[[V2:.*]] = test_inst outs %[[DEAD]] ins %[[DEAD]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[LIVE]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @dead_write_interference {
  %live_through = alloca : !amdgcn.vgpr<?>
  %dead_target = alloca : !amdgcn.vgpr<?>
  %src = alloca : !amdgcn.vgpr<?>
  %0 = test_inst outs %live_through : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %1 = test_inst outs %src : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>

  // Dead write: result %2 is never used.
  %2 = test_inst outs %dead_target ins %src : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>

  // %live_through is used here, so it's live through the dead write above.
  test_inst ins %live_through : (!amdgcn.vgpr<?>) -> ()
  end_kernel
}

// -----

// CHECK-LABEL:   amdgcn.kernel @dead_copy_interference {
// CHECK-DAG:       %[[A:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[B:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[SRC:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[V0:.*]] = test_inst outs %[[A]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[V1:.*]] = test_inst outs %[[SRC]] : (!amdgcn.vgpr<2>) -> !amdgcn.vgpr<2>
// CHECK:           %[[V2:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[B]], %[[SRC]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:           %[[V3:.*]] = test_inst outs %[[B]] ins %[[SRC]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:           %[[V4:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[B]], %[[A]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[B]] : (!amdgcn.vgpr<1>) -> ()
// CHECK:           end_kernel
// CHECK:         }
amdgcn.kernel @dead_copy_interference {
  %a = alloca : !amdgcn.vgpr<?>
  %b = alloca : !amdgcn.vgpr<?>
  %c = alloca : !amdgcn.vgpr<?>
  %src = alloca : !amdgcn.vgpr<?>
  %0 = test_inst outs %a : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
  %1 = test_inst outs %src : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>

  // Dead copy: writes to %b, but %b is overwritten at %4 before being read.
  %2 = lsir.copy %b, %src : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>

  // Use %src in another instruction.
  %3 = test_inst outs %c ins %src : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>

  // Overwrite %b with %a's value.
  %4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %b, %a : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>

  // %b is used here, so it's live through the dead copy above.
  test_inst ins %b : (!amdgcn.vgpr<?>) -> ()

  end_kernel
}
