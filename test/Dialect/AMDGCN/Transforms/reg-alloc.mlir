// RUN: aster-opt %s --amdgcn-register-allocation --cse --aster-disable-verifiers --aster-suppress-disabled-verifier-warning | FileCheck %s

// CHECK-LABEL:   amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @range_allocations {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr<4>
// CHECK:             %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<5>
// CHECK:             %[[VAL_6:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:             %[[VAL_7:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:             %[[VAL_8:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
// CHECK:             %[[VAL_9:.*]] = test_inst outs %[[VAL_6]] ins %[[VAL_8]] : (!amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr_range<[4 : 6]>) -> !amdgcn.vgpr_range<[0 : 2]>
// CHECK:             test_inst ins %[[VAL_7]] : (!amdgcn.vgpr_range<[0 : 4]>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @range_allocations {
    // Test register allocation on range allocations.
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    %7 = alloca : !amdgcn.vgpr
    %8 = make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
    %9 = make_register_range %0, %1, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %10 = make_register_range %6, %7 : !amdgcn.vgpr, !amdgcn.vgpr
    %11 = test_inst outs %8 ins %10 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 2]>
    // The main point in this test is that %8 introduces a conflict with %10,
    // and as consequence %9. Therefore %0-%3 and %6-%7 all must have different
    // colors.
    test_inst ins %9 : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    end_kernel
  }
// CHECK-LABEL:     kernel @no_interference_mixed {
// CHECK:             %[[VAL_10:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_11:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_12:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_13:.*]] = test_inst outs %[[VAL_10]] ins %[[VAL_11]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_14:.*]] = test_inst outs %[[VAL_10]] ins %[[VAL_12]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interference_mixed {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    // There's no interference, so check registers are reused.
    %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
// CHECK-LABEL:     kernel @interference_mixed_with_reuse {
// CHECK:             %[[VAL_15:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_16:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_17:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_18:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_19:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[VAL_20:.*]] = test_inst outs %[[VAL_15]] ins %[[VAL_17]], %[[VAL_16]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_21:.*]] = test_inst outs %[[VAL_16]] ins %[[VAL_18]], %[[VAL_19]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[VAL_20]], %[[VAL_21]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_with_reuse {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // In this case almost everyone interferes, however, %4 and %1 can share
    // registers as %4 is dead after %6.
    test_inst ins %6, %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
// CHECK-LABEL:     kernel @interference_mixed_all_live {
// CHECK:             %[[VAL_22:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_23:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_24:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_25:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_26:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[VAL_27:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[VAL_28:.*]] = test_inst outs %[[VAL_22]] ins %[[VAL_24]], %[[VAL_26]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_29:.*]] = test_inst outs %[[VAL_23]] ins %[[VAL_25]], %[[VAL_27]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[VAL_28]], %[[VAL_29]], %[[VAL_26]], %[[VAL_27]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_all_live {
    // Everyone interferes, therefore allocate every allocation in a different register.
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %6, %7, %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
// CHECK-LABEL:     func.func private @rand() -> i1
  func.func private @rand() -> i1
// CHECK-LABEL:     kernel @no_interference_cf {
// CHECK:             %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_30:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_31:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_32:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_33:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_34:.*]] = test_inst outs %[[VAL_30]] ins %[[VAL_32]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_35:.*]] = test_inst outs %[[VAL_31]] ins %[[VAL_33]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_36:.*]] = test_inst outs %[[VAL_31]] ins %[[VAL_34]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_37:.*]] = test_inst outs %[[VAL_30]] ins %[[VAL_35]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interference_cf {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    // While %7, %8 don't interfere in this block, they interfere with %9, %10
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%8, %2) because they are dead.
    %9 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%7, %1) because they are dead.
    %10 = test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    end_kernel
  }
// CHECK-LABEL:     kernel @interference_cf {
// CHECK:             %[[CALL_1:.*]] = func.call @rand() : () -> i1
// CHECK:             %[[VAL_38:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_39:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_40:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_41:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_42:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[VAL_43:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[VAL_44:.*]] = test_inst outs %[[VAL_38]] ins %[[VAL_40]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_45:.*]] = test_inst outs %[[VAL_39]] ins %[[VAL_41]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<1>
// CHECK:             cf.cond_br %[[CALL_1]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[VAL_46:.*]] = test_inst outs %[[VAL_42]] ins %[[VAL_44]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<2>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[VAL_47:.*]] = test_inst outs %[[VAL_43]] ins %[[VAL_45]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<3>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             test_inst ins %[[VAL_42]], %[[VAL_43]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_cf {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %9 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    %10 = test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    // Unlike in the previous case, we can't reuse the allocation of (%5, %6)
    // because they are live.
    test_inst ins %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
// CHECK-LABEL:     kernel @existing_regs {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK:             %[[VAL_5:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[VAL_6:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[VAL_7:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:             %[[VAL_8:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:             %[[VAL_9:.*]] = alloca : !amdgcn.sgpr<8>
// CHECK:             %[[VAL_10:.*]] = alloca : !amdgcn.sgpr<9>
// CHECK:             %[[VAL_11:.*]] = alloca : !amdgcn.sgpr<10>
// CHECK:             %[[VAL_12:.*]] = alloca : !amdgcn.sgpr<11>
// CHECK:             %[[VAL_13:.*]] = make_register_range %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : !amdgcn.sgpr<8>, !amdgcn.sgpr<9>, !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
// CHECK:             test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_13]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<5>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.sgpr<1>, !amdgcn.sgpr<3>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @existing_regs {
    %0 = alloca : !amdgcn.vgpr<0>
    %1 = alloca : !amdgcn.vgpr<1>
    %2 = alloca : !amdgcn.sgpr<0>
    %3 = alloca : !amdgcn.sgpr<2>
    %4 = alloca : !amdgcn.sgpr<5>
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    %7 = alloca : !amdgcn.sgpr
    %8 = alloca : !amdgcn.sgpr
    %9 = alloca : !amdgcn.sgpr
    %10 = alloca : !amdgcn.sgpr
    %11 = alloca : !amdgcn.sgpr
    %12 = alloca : !amdgcn.sgpr
    %13 = make_register_range %9, %10, %11, %12 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    // Test that existing registers are respected
    test_inst ins %0, %1, %2, %3, %4, %5, %6, %7, %8, %13 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<5>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }

// Test that values used by make_register_range are kept live, creating
// interference with intermediate definitions.
// CHECK-LABEL:   kernel @make_range_liveness {
// CHECK:           %[[A:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:           %[[B:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[RES_A:.*]] = test_inst outs %[[A]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[RES_B:.*]] = test_inst outs %[[B]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// Intermediate values must NOT reuse vgpr<0> or vgpr<1> since %RES_A and %RES_B are still live
// CHECK:           %[[TMP1:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[TMP2:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[INT:.*]] = test_inst outs %[[TMP1]] ins %[[TMP2]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<2>
// CHECK:           %[[RANGE:.*]] = make_register_range %[[RES_A]], %[[RES_B]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[RANGE]], %[[INT]] : (!amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr<2>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @make_range_liveness {
    %a = alloca : !amdgcn.vgpr
    %b = alloca : !amdgcn.vgpr
    // Define values that will be used in make_register_range
    %res_a = test_inst outs %a : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %res_b = test_inst outs %b : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // Intermediate computation - these allocas must not reuse %a or %b's registers
    // because %res_a and %res_b are still live (used by make_register_range below)
    %tmp1 = alloca : !amdgcn.vgpr
    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate = test_inst outs %tmp1 ins %tmp2 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // Create range from values defined earlier - this use should keep %res_a, %res_b live
    %range = make_register_range %res_a, %res_b : !amdgcn.vgpr, !amdgcn.vgpr
    // Use both the range and the intermediate result
    test_inst ins %range, %intermediate : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr) -> ()
    end_kernel
  }

// CHECK-LABEL:   kernel @reg_interference {
// CHECK:           %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:           %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK:           %[[VAL_2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:           test_inst ins %[[VAL_1]], %[[VAL_2]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<2>) -> ()
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK:           %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           test_inst ins %[[VAL_0]], %[[VAL_3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @reg_interference {
    %0 = alloca : !amdgcn.sgpr
    %1 = alloca : !amdgcn.sgpr
    test_inst ins %0, %1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    test_inst ins %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.reg_interference %0, %2, %3 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.sgpr
    test_inst ins %4, %5 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    %6 = alloca : !amdgcn.sgpr
    %7 = alloca : !amdgcn.sgpr
    test_inst ins %6, %7 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.reg_interference %4, %1, %3, %7 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    end_kernel
  }
}
