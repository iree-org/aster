// RUN: aster-opt %s --test-amdgcn-interference-analysis=optimize=true --split-input-file 2>&1 | FileCheck %s

amdgcn.module @reg_alloc target = <gfx942> isa = <cdna3> {
// CHECK-LABEL: Function: coalescing_load
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %0"];
// CHECK:   1 [label="1, %1"];
// CHECK:   2 [label="2, %2"];
// CHECK:   3 [label="3, %3"];
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %0"];
// CHECK:   1 [label="1, %1"];
// CHECK:   2 [label="2, %2"];
// CHECK: }
  kernel @coalescing_load {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    %4 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %token = load global_load_dword dest %2 addr %4 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %3, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %3 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }

// CHECK-LABEL: Function: coalescing_all
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %0"];
// CHECK:   1 [label="1, %1"];
// CHECK:   2 [label="2, %2"];
// CHECK:   3 [label="3, %3"];
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0, 1, 2, 3]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %0"];
// CHECK: }
  kernel @coalescing_all {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %2, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %3, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %3 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }

// CHECK-LABEL: Function: cannot_coalesce
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %0"];
// CHECK:   1 [label="1, %1"];
// CHECK:   2 [label="2, %2"];
// CHECK:   3 [label="3, %3"];
// CHECK:   0 -- 1;
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   2 -- 3;
// CHECK: }
// CHECK-NOT: EquivalenceClasses {
  kernel @cannot_coalesce {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %2, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %3, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %0, %1, %2, %3 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL:  Function: incompatible_range_offsets
// CHECK-NOT: EquivalenceClasses
func.func @incompatible_range_offsets() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.alloca : !amdgcn.vgpr<?>
  %8 = amdgcn.alloca : !amdgcn.vgpr<?>
  %9 = amdgcn.alloca : !amdgcn.vgpr<?>
  %10 = amdgcn.make_register_range %0, %1, %2, %3, %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %11 = amdgcn.make_register_range %6, %7, %8, %9 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %10, %11 : (!amdgcn.vgpr<[? : ? + 6]>, !amdgcn.vgpr<[? : ? + 4]>) -> ()
  %12 = amdgcn.make_register_range %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %13 = amdgcn.make_register_range %8, %9 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  // The alignments are not compatible, so the ranges are not coalesced.
  lsir.copy %12, %13 : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
  return
}

// CHECK-LABEL:  Function: compatible_range_offsets_0
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    4 [label="4, %{{.*}}"];
// CHECK:    5 [label="5, %{{.*}}"];
// CHECK:    6 [label="6, %{{.*}}"];
// CHECK:    7 [label="7, %{{.*}}"];
// CHECK:    2 -- 3;
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0, 4]
// CHECK:    [1, 5]
// CHECK:    [2, 6]
// CHECK:    [3, 7]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    2 -- 3;
// CHECK:  }
func.func @compatible_range_offsets_0() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.alloca : !amdgcn.vgpr<?>
  %6 = amdgcn.alloca : !amdgcn.vgpr<?>
  %7 = amdgcn.alloca : !amdgcn.vgpr<?>
  %8 = amdgcn.make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %9 = amdgcn.make_register_range %4, %5, %6, %7 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %8, %9 : (!amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[? : ? + 4]>) -> ()
  %10 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %11 = amdgcn.make_register_range %6, %7 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %10, %11 : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
  return
}

// CHECK-LABEL:  Function: compatible_ranges_1
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    4 [label="4, %{{.*}}"];
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0, 4]
// CHECK:    [1]
// CHECK:    [2]
// CHECK:    [3]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:  }
func.func @compatible_ranges_1() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %5, %4 : (!amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %0, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// CHECK-LABEL:  Function: compatible_ranges_2
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    4 [label="4, %{{.*}}"];
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0]
// CHECK:    [1, 4]
// CHECK:    [2]
// CHECK:    [3]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:  }
func.func @compatible_ranges_2() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %5, %4 : (!amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %1, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// CHECK-LABEL:  Function: compatible_ranges_3
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    4 [label="4, %{{.*}}"];
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0]
// CHECK:    [1]
// CHECK:    [2, 4]
// CHECK:    [3]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:  }
func.func @compatible_ranges_3() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %5, %4 : (!amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %2, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// CHECK-LABEL:  Function: compatible_ranges_4
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    4 [label="4, %{{.*}}"];
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0]
// CHECK:    [1]
// CHECK:    [2]
// CHECK:    [3, 4]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:  }
func.func @compatible_ranges_4() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.alloca : !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %5, %4 : (!amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<?>) -> ()
  lsir.copy %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}

// -----

// CHECK-LABEL:  Function: allocated_regs
// CHECK-NOT: EquivalenceClasses
func.func @allocated_regs() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<2>
  lsir.copy %1, %0 : !amdgcn.vgpr<2>, !amdgcn.vgpr<1>
  return
}

// CHECK-LABEL:  Function: semi_allocated_regs
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0, 1]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:  }
func.func @semi_allocated_regs() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %1, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<1>
  return
}

// CHECK-LABEL:  Function: different_reg_kinds
// CHECK-NOT: EquivalenceClasses
func.func @different_reg_kinds() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.sgpr<?>
  lsir.copy %0, %1 : !amdgcn.vgpr<?>, !amdgcn.sgpr<?>
  return
}

// CHECK-LABEL:  Function: disassembled_copies
// CHECK:  graph RegisterInterference {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    2 [label="2, %{{.*}}"];
// CHECK:    3 [label="3, %{{.*}}"];
// CHECK:    0 -- 3;
// CHECK:  }
// CHECK:  EquivalenceClasses {
// CHECK:    [0, 2]
// CHECK:    [1, 3]
// CHECK:  }
// CHECK:  graph RegisterInterferenceQuotient {
// CHECK:    0 [label="0, %{{.*}}"];
// CHECK:    1 [label="1, %{{.*}}"];
// CHECK:    0 -- 1;
// CHECK:  }
func.func @disassembled_copies() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.alloca : !amdgcn.vgpr<?>
  %4 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %5 = amdgcn.make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %4, %5 : (!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>) -> ()
  lsir.copy %0, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  lsir.copy %1, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return
}
