// RUN: aster-opt %s --cse --amdgcn-register-allocation --cse --aster-disable-verifiers --aster-suppress-disabled-verifier-warning --split-input-file | FileCheck %s

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @range_allocations {
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK-DAG:         %[[V4:.*]] = alloca : !amdgcn.vgpr<4>
// CHECK-DAG:         %[[V5:.*]] = alloca : !amdgcn.vgpr<5>
// CHECK:             %[[A:.*]] = make_register_range %[[V0]], %[[V1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:             %[[B:.*]] = make_register_range %[[V0]], %[[V1]], %[[V2]], %[[V3]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:             %[[C:.*]] = make_register_range %[[V4]], %[[V5]] : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>
// CHECK:             %[[E:.*]] = test_inst outs %[[A]] ins %[[C]] : (!amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr_range<[4 : 6]>) -> !amdgcn.vgpr_range<[0 : 2]>
// CHECK:             test_inst ins %[[B]] : (!amdgcn.vgpr_range<[0 : 4]>) -> ()
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
    %a = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %8 = make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
    %9 = make_register_range %0, %1, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %10 = make_register_range %6, %7 : !amdgcn.vgpr, !amdgcn.vgpr
    %11 = test_inst outs %8 ins %10 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 2]>
    // The main point in this test is that %8 introduces a conflict with %10,
    // and as consequence %9. Therefore %0-%3 and %6-%7 all must have different
    // colors.
    test_inst ins %9 : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    test_inst ins %a, %b : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @no_interference_mixed_undef_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[C0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[A:.*]] = test_inst outs %[[C0]] ins %[[S0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[B:.*]] = test_inst outs %[[C0]] ins %[[S0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interference_mixed_undef_values {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    // There's no interference, so check registers are reused.
    %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// CHECK-LABEL:     kernel @no_interferencemixed_with_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[V1:.*]] = test_inst outs %[[S0]] : (!amdgcn.sgpr<0>) -> !amdgcn.sgpr<0>
// CHECK:             %[[V2:.*]] = test_inst outs %[[S1]] : (!amdgcn.sgpr<1>) -> !amdgcn.sgpr<1>
// CHECK:             %[[VAL_13:.*]] = test_inst outs %[[V0]] ins %[[V1]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[VAL_14:.*]] = test_inst outs %[[V0]] ins %[[V2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interferencemixed_with_values {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %a = test_inst outs %2 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %b = test_inst outs %3 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    // There's no interference, so check registers are reused.
    %4 = test_inst outs %0 ins %a : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %5 = test_inst outs %1 ins %b : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// CHECK-LABEL:     kernel @interference_mixed_with_reuse_undef_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:             %[[V3:.*]] = test_inst outs %[[V0]] ins %[[S0]], %[[V0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[V4:.*]] = test_inst outs %[[V1]] ins %[[S0]], %[[V2]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[V3]], %[[V4]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_with_reuse_undef_values {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // In this case almost everyone interferes, but there are no live values so
    // we can do some aggressive reuse.
    test_inst ins %6, %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// CHECK-LABEL:     kernel @interference_mixed_with_reuse_with_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// NOTE: %b#0 is never used but it still gets a unique register to avoid clobbering.
// The proper way to optimize this is to run DCE before register allocation.
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[AA:.*]]:2 = test_inst outs %[[S0]], %[[S1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:             %[[BB:.*]]:2 = test_inst outs %[[V2]], %[[V3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[AA]]#0, %[[AA]]#1 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[S1]], %[[BB]]#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[C]], %[[D]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_with_reuse_with_values {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %a:2 = test_inst outs %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %b:2 = test_inst outs %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
    %6 = test_inst outs %0 ins %a#0, %a#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3, %b#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // In this case almost everyone interferes, however, %4 and %1 can share
    // registers as the value flowing through %4 is dead after %6.
    test_inst ins %6, %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @interference_mixed_all_live_undef_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[S0]], %[[V2]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[S0]], %[[V3]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[C]], %[[D]], %[[V2]], %[[V3]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_all_live_undef_values {
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
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @interference_mixed_all_live_with_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[AA:.*]]:2 = test_inst outs %[[S0]], %[[S1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:             %[[BB:.*]]:2 = test_inst outs %[[V2]], %[[V3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[AA]]#0, %[[BB]]#0 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[AA]]#1, %[[BB]]#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             test_inst ins %[[C]], %[[D]], %[[BB]]#0, %[[BB]]#1 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_mixed_all_live_with_values {
    // Everyone interferes, therefore allocate every allocation in a different register.
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %a:2 = test_inst outs %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %b:2 = test_inst outs %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
    %6 = test_inst outs %0 ins %a#0, %b#0 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %a#1, %b#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %6, %7, %b#0, %b#1 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     func.func private @rand() -> i1
  func.func private @rand() -> i1

// CHECK-LABEL:     kernel @no_interference_cf_undef_values {
// CHECK-DAG:         %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[S0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[S0]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<1>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[E:.*]] = test_inst outs %[[V0]] ins %[[C]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[F:.*]] = test_inst outs %[[V0]] ins %[[D]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interference_cf_undef_values {
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
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     func.func private @rand() -> i1
  func.func private @rand() -> i1

// CHECK-LABEL:     kernel @no_interference_cf_with_values {
// CHECK-DAG:         %[[CALL_0:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[AA:.*]]:2 = test_inst outs %[[S0]], %[[S1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:             %[[BB:.*]]:2 = test_inst outs %[[V2]], %[[V3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[AA]]#0, %[[BB]]#0 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[AA]]#1, %[[BB]]#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             cf.cond_br %[[CALL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[E:.*]] = test_inst outs %[[V1]] ins %[[C]], %[[BB]]#0 : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<1>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[F:.*]] = test_inst outs %[[V0]] ins %[[D]], %[[BB]]#1 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<0>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             end_kernel
// CHECK:           }
  kernel @no_interference_cf_with_values {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    // While %7, %8 don't interfere in this block, they interfere with %9, %10
    %a:2 = test_inst outs %3, %4 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %b:2 = test_inst outs %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
    %7 = test_inst outs %1 ins %a#0, %b#0 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %a#1, %b#1 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%8, %2) because they are dead.
    %9 = test_inst outs %2 ins %7, %b#0 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%7, %1) because they are dead.
    %10 = test_inst outs %1 ins %8, %b#1 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     func.func private @rand() -> i1
  func.func private @rand() -> i1

// CHECK-LABEL:     kernel @interference_cf_with_values {
// CHECK:             %[[CALL_1:.*]] = func.call @rand() : () -> i1
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[S0]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[S0]] : (!amdgcn.vgpr<1>, !amdgcn.sgpr<0>) -> !amdgcn.vgpr<1>
// CHECK:             cf.cond_br %[[CALL_1]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             %[[E:.*]] = test_inst outs %[[V2]] ins %[[C]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<2>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb2:
// CHECK:             %[[F:.*]] = test_inst outs %[[V3]] ins %[[D]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<3>
// CHECK:             cf.br ^bb3
// CHECK:           ^bb3:
// CHECK:             test_inst ins %[[V2]], %[[V3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @interference_cf_with_values {
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
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @existing_regs_undef_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[S2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:         %[[S3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:         %[[S5:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:             %[[RANGE:.*]] = make_register_range %[[S0]], %[[S1]], %[[S2]], %[[S3]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
// CHECK:             test_inst ins %[[V0]], %[[V1]], %[[S0]], %[[S2]], %[[S5]], %[[V0]], %[[V0]], %[[S0]], %[[S0]], %[[RANGE]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr<5>, !amdgcn.vgpr<0>, !amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.sgpr<0>, !amdgcn.sgpr_range<[0 : 4]>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @existing_regs_undef_values {
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
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @existing_regs_with_values {
// CHECK-DAG:         %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:         %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:         %[[S10:.*]] = alloca : !amdgcn.sgpr<10>
// CHECK-DAG:         %[[S11:.*]] = alloca : !amdgcn.sgpr<11>
// CHECK-DAG:         %[[S2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:         %[[S3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:         %[[S5:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK-DAG:         %[[S8:.*]] = alloca : !amdgcn.sgpr<8>
// CHECK-DAG:         %[[S9:.*]] = alloca : !amdgcn.sgpr<9>
// CHECK-DAG:         %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:         %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:         %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:         %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:             %[[AA:.*]]:2 = test_inst outs %[[S0]], %[[S2]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
// CHECK:             %[[BB:.*]]:2 = test_inst outs %[[V2]], %[[V3]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>)
// CHECK:             %[[CC:.*]]:2 = test_inst outs %[[S1]], %[[S3]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<3>) -> (!amdgcn.sgpr<1>, !amdgcn.sgpr<3>)
// CHECK:             %[[DD:.*]]:2 = test_inst outs %[[S8]], %[[S9]] : (!amdgcn.sgpr<8>, !amdgcn.sgpr<9>) -> (!amdgcn.sgpr<8>, !amdgcn.sgpr<9>)
// CHECK:             %[[EE:.*]]:2 = test_inst outs %[[S10]], %[[S11]] : (!amdgcn.sgpr<10>, !amdgcn.sgpr<11>) -> (!amdgcn.sgpr<10>, !amdgcn.sgpr<11>)
// CHECK:             %[[C:.*]] = test_inst outs %[[V0]] ins %[[AA]]#0, %[[BB]]#0 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>
// CHECK:             %[[D:.*]] = test_inst outs %[[V1]] ins %[[AA]]#1, %[[BB]]#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr<3>) -> !amdgcn.vgpr<1>
// CHECK:             %[[RANGE:.*]] = make_register_range %[[DD]]#0, %[[DD]]#1, %[[EE]]#0, %[[EE]]#1 : !amdgcn.sgpr<8>, !amdgcn.sgpr<9>, !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
// CHECK:             test_inst ins %[[C]], %[[D]], %[[CC]]#0, %[[CC]]#1, %[[S5]], %[[BB]]#0, %[[BB]]#1, %[[AA]]#0, %[[AA]]#1, %[[RANGE]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr<1>, !amdgcn.sgpr<3>, !amdgcn.sgpr<5>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
// CHECK:             end_kernel
// CHECK:           }
  kernel @existing_regs_with_values {
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
    %a:2 = test_inst outs %2, %3 : (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
    %b:2 = test_inst outs %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
    %c:2 = test_inst outs %7, %8 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %d:2 = test_inst outs %9, %10 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %e:2 = test_inst outs %11, %12 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    %13 = test_inst outs %0 ins %a#0, %b#0 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr) -> !amdgcn.vgpr<0>
    %14 = test_inst outs %1 ins %a#1, %b#1 : (!amdgcn.vgpr<1>, !amdgcn.sgpr<2>, !amdgcn.vgpr) -> !amdgcn.vgpr<1>
    %15 = make_register_range %d#0, %d#1, %e#0, %e#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    // Test that existing registers are respected
    test_inst ins %13, %14, %c#0, %c#1, %4, %b#0, %b#1, %a#0, %a#1, %15 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr<5>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, !amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// Test that values used by make_register_range are kept live, creating
// interference with intermediate definitions.
// CHECK-LABEL:   kernel @test_make_range_liveness_1 {
// CHECK-DAG:       %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:       %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:       %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
//
// CHECK:           %[[RES_A:.*]] = test_inst outs %[[V0]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[RES_B:.*]] = test_inst outs %[[V1]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
//
// Intermediate values must NOT reuse vgpr<0> or vgpr<1> since %RES_A and %RES_B are still live
// CHECK:           %[[INT_0:.*]] = test_inst outs %[[V2]] ins %[[RES_B]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK:           %[[INT_1:.*]] = test_inst outs %[[V3]] ins %[[INT_0]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           %[[RANGE:.*]] = make_register_range %[[RES_A]], %[[RES_B]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[RANGE]], %[[INT_0]] : (!amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr<2>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @test_make_range_liveness_1 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %tmp1 = alloca : !amdgcn.vgpr
    %tmp2 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    // intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    // Use at the same time as the range.
    test_inst ins %range, %intermediate_0 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// Test that values used by make_register_range are kept live, creating
// interference with intermediate definitions.
// CHECK-LABEL:   kernel @test_make_range_liveness_2 {
// CHECK-DAG:       %[[A:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[B:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[RES_A:.*]] = test_inst outs %[[A]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[RES_B:.*]] = test_inst outs %[[B]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// Intermediate values must NOT reuse vgpr<0> or vgpr<1> since %RES_A and %RES_B are still live
// CHECK-DAG:       %[[TMP1:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[INT_0:.*]] = test_inst outs %[[TMP1]] ins %[[RES_B]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK-DAG:       %[[TMP2:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[INT_1:.*]] = test_inst outs %[[TMP2]] ins %[[INT_0]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           test_inst ins %[[INT_0]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           %[[RANGE:.*]] = make_register_range %[[RES_A]], %[[RES_B]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[RANGE]] : (!amdgcn.vgpr_range<[0 : 2]>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @test_make_range_liveness_2 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    %tmp1 = alloca : !amdgcn.vgpr
    // intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    // Use before the range.
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// Test that values used by make_register_range are kept live, creating
// interference with intermediate definitions.
// CHECK-LABEL:   kernel @test_make_range_liveness_3 {
// CHECK-DAG:       %[[A:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:       %[[B:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:           %[[RES_A:.*]] = test_inst outs %[[A]] : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<0>
// CHECK:           %[[RES_B:.*]] = test_inst outs %[[B]] : (!amdgcn.vgpr<1>) -> !amdgcn.vgpr<1>
// Intermediate values must NOT reuse vgpr<0> or vgpr<1> since %RES_A and %RES_B are still live
// CHECK-DAG:       %[[TMP1:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:           %[[INT_0:.*]] = test_inst outs %[[TMP1]] ins %[[RES_B]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
// CHECK-DAG:       %[[TMP2:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK:           %[[INT_1:.*]] = test_inst outs %[[TMP2]] ins %[[INT_0]] : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<3>
// CHECK:           %[[RANGE:.*]] = make_register_range %[[RES_A]], %[[RES_B]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           test_inst ins %[[RANGE]] : (!amdgcn.vgpr_range<[0 : 2]>) -> ()
// CHECK:           test_inst ins %[[INT_0]] : (!amdgcn.vgpr<2>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @test_make_range_liveness_3 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    %tmp1 = alloca : !amdgcn.vgpr
    // intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    // Use after the range.
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// With CSE and **live after** interference graph, many undef allocas get merged.
// CHECK-LABEL:   kernel @reg_interference_undef_values {
// CHECK-DAG:       %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:           test_inst ins %[[S0]], %[[S0]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<0>) -> ()
// CHECK-DAG:       %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:       %[[S2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:           test_inst ins %[[S1]], %[[S2]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<2>) -> ()
// CHECK:           test_inst ins %[[S1]], %[[S0]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<0>) -> ()
// CHECK-DAG:       %[[S3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           test_inst ins %[[S0]], %[[S3]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<3>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @reg_interference_undef_values {
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

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

// CHECK-LABEL:   kernel @reg_interference_with_values {
// CHECK-DAG:       %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:       %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:           %[[AA:.*]]:2 = test_inst outs %[[S0]], %[[S1]] : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:           test_inst ins %[[AA]]#0, %[[AA]]#1 : (!amdgcn.sgpr<0>, !amdgcn.sgpr<1>) -> ()
// CHECK-DAG:       %[[S2:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:       %[[S3:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:           %[[BB:.*]]:2 = test_inst outs %[[S2]], %[[S3]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<3>) -> (!amdgcn.sgpr<2>, !amdgcn.sgpr<3>)
// CHECK:           test_inst ins %[[AA]]#1, %[[BB]]#0 : (!amdgcn.sgpr<1>, !amdgcn.sgpr<2>) -> ()
// CHECK-DAG:       %[[S4:.*]] = alloca : !amdgcn.sgpr<4>
// CHECK:           %[[CC:.*]]:2 = test_inst outs %[[S2]], %[[S4]] : (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>) -> (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>)
// CHECK:           test_inst ins %[[CC]]#0, %[[CC]]#1 : (!amdgcn.sgpr<2>, !amdgcn.sgpr<4>) -> ()
// CHECK:           %[[S5:.*]] = alloca : !amdgcn.sgpr<5>
// CHECK:           %[[DD:.*]]:2 = test_inst outs %[[S1]], %[[S5]] : (!amdgcn.sgpr<1>, !amdgcn.sgpr<5>) -> (!amdgcn.sgpr<1>, !amdgcn.sgpr<5>)
// CHECK:           test_inst ins %[[DD]]#0, %[[DD]]#1 : (!amdgcn.sgpr<1>, !amdgcn.sgpr<5>) -> ()
// CHECK:           test_inst ins %[[AA]]#0, %[[CC]]#1 : (!amdgcn.sgpr<0>, !amdgcn.sgpr<4>) -> ()
// CHECK:           end_kernel
// CHECK:         }
  amdgcn.kernel @reg_interference_with_values {
    %0 = alloca : !amdgcn.sgpr
    %1 = alloca : !amdgcn.sgpr
    %a:2 = test_inst outs %0, %1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %a#0, %a#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %b:2 = test_inst outs %2, %3 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %a#1, %b#0 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.reg_interference %a#0, %b#0, %b#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.sgpr
    %c:2 = test_inst outs %4, %5 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %c#0, %c#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    %6 = alloca : !amdgcn.sgpr
    %7 = alloca : !amdgcn.sgpr
    %d:2 = test_inst outs %6, %7 : (!amdgcn.sgpr, !amdgcn.sgpr) -> (!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %d#0, %d#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.reg_interference %c#0, %a#1, %b#1, %d#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %a#0, %c#1 : (!amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:   amdgcn.module
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

  amdgcn.kernel @test_index_bxmxnxk arguments <[#amdgcn.buffer_arg<address_space = generic>, #amdgcn.block_dim_arg<x>]> {
    %c42 = arith.constant 42 : i32
    // Preallocated alloca establish a values we must not clobber.
    %1 = alloca : !amdgcn.sgpr<2>
    %2 = alloca : !amdgcn.sgpr<0>
    %3 = alloca : !amdgcn.sgpr<1>

    // Load dword from kernel argument [s0, s1]
    %4 = make_register_range %2, %3 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %5 = alloca : !amdgcn.sgpr
    %result, %token = load s_load_dword dest %5 addr %4 offset c(%c42)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0

    // And on the loaded result.
    %6 = alloca : !amdgcn.sgpr
    %7 = sop2 s_and_b32 outs %6 ins %result, %c42 : !amdgcn.sgpr, !amdgcn.sgpr, i32

    // Load dwordx2 from kernel argument [s0, s1]
    %8 = alloca : !amdgcn.sgpr<0>
    %9 = alloca : !amdgcn.sgpr<1>
    %10 = make_register_range %8, %9 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %11 = alloca : !amdgcn.sgpr
    %12 = alloca : !amdgcn.sgpr
    %13 = make_register_range %11, %12 : !amdgcn.sgpr, !amdgcn.sgpr
    %result_0, %token_1 = load s_load_dwordx2 dest %13 addr %10 offset c(%c42)
      : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>

    // %7 must not clobber !amdgcn.sgpr<0>
    // CHECK-NOT: test_inst ins {{.*}} !amdgcn.sgpr<0>
    test_inst ins %1, %7 : (!amdgcn.sgpr<2>, !amdgcn.sgpr) -> ()

    end_kernel
  }
}
