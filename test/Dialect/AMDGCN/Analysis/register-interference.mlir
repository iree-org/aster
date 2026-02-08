// RUN: aster-opt %s --test-amdgcn-interference-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: no_interference
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @no_interference {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    end_kernel
  }
}

// -----
// CHECK-LABEL: basic_interference
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   0 -- 1;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @basic_interference {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: three_way_interference
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK:   1 -- 2;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @three_way_interference {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %4 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %5 = test_inst outs %2 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %0, %1, %2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: no_cross_type_interference
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @no_cross_type_interference {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.sgpr<?>
    %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %3 = test_inst outs %1 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
    test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: reg_interference_op
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK:   1 -- 2;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @reg_interference_op {
    %0 = alloca : !amdgcn.sgpr<?>
    %1 = alloca : !amdgcn.sgpr<?>
    %2 = alloca : !amdgcn.sgpr<?>
    %3 = test_inst outs %0 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
    %4 = test_inst outs %1 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
    %5 = test_inst outs %2 : (!amdgcn.sgpr<?>) -> !amdgcn.sgpr<?>
    reg_interference %0, %1, %2 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
    end_kernel
  }
}

// -----
// CHECK-LABEL: partial_interference
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   0 -- 1;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @partial_interference {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %4 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %5 = test_inst outs %2 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %2 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: diamond_cf
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @diamond_cf {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // CHECK: pred: ^bb0
    %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
    cf.br ^bb3
  ^bb2:  // CHECK: pred: ^bb0
    %4 = test_inst outs %2 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %2 : (!amdgcn.vgpr<?>) -> ()
    cf.br ^bb3
  ^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
    end_kernel
  }
}

// -----
// CHECK-LABEL: live_across_diamond
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @live_across_diamond {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    %4 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // CHECK: pred: ^bb0
    %5 = test_inst outs %2 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %2 : (!amdgcn.vgpr<?>) -> ()
    cf.br ^bb3
  ^bb2:  // CHECK: pred: ^bb0
    %6 = test_inst outs %3 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %3 : (!amdgcn.vgpr<?>) -> ()
    cf.br ^bb3
  ^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
    test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: sequential_use
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @sequential_use {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
    %3 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %1 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: many_overlapping
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   3 [label="3
// CHECK:   4 [label="4
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   2 -- 3;
// CHECK:   2 -- 4;
// CHECK:   3 -- 4;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @many_overlapping {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    %4 = alloca : !amdgcn.vgpr<?>
    %5 = test_inst outs %0 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %6 = test_inst outs %1 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %7 = test_inst outs %2 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %8 = test_inst outs %3 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %9 = test_inst outs %4 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    test_inst ins %0, %1, %2, %3, %4 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: phi_coalescing_2
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   3 [label="3
// CHECK:   4 [label="4
// CHECK:   5 [label="5
// CHECK:   6 [label="6
// CHECK:   7 [label="7
// CHECK:   8 [label="8
// CHECK:   0 -- 1;
// CHECK:   2 -- 3;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_2 {
    %c0_i32 = arith.constant 0 : i32
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.sgpr<?>
    %3 = alloca : !amdgcn.sgpr<?>
    %4 = alloca : !amdgcn.vgpr<?>
    %5 = alloca : !amdgcn.vgpr<?>
    %6 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
    %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
    %9 = alloca : !amdgcn.vgpr<?>
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // CHECK: pred: ^bb0
    %10 = test_inst outs %4 ins %0 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %11 = alloca : !amdgcn.vgpr<?>
    %12 = test_inst outs %11 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %13 = lsir.copy %9, %11 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb2:  // CHECK: pred: ^bb0
    %14 = test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %15 = alloca : !amdgcn.vgpr<?>
    %16 = test_inst outs %15 : (!amdgcn.vgpr<?>) -> !amdgcn.vgpr<?>
    %17 = lsir.copy %9, %15 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
    test_inst ins %9 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL: phi_coalescing_3
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0
// CHECK:   1 [label="1
// CHECK:   2 [label="2
// CHECK:   3 [label="3
// CHECK:   4 [label="4
// CHECK:   0 -- 1;
// CHECK:   0 -- 4;
// CHECK:   1 -- 4;
// CHECK:   2 -- 3;
// CHECK: }
amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_3 {
    %c0_i32 = arith.constant 0 : i32
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.sgpr<?>
    %3 = alloca : !amdgcn.sgpr<?>
    %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
    %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> !amdgcn.vgpr<?>
    %6 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
    %7 = alloca : !amdgcn.vgpr<?>
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // CHECK: pred: ^bb0
    %8 = lsir.copy %7, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb2:  // CHECK: pred: ^bb0
    %9 = lsir.copy %7, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb3:  // CHECK: 2 preds: ^bb1, ^bb2
    test_inst ins %7, %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}
