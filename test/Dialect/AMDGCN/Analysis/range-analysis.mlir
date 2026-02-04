// RUN: aster-opt %s --test-range-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: simple_range

// Simple test: single range with two registers
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK:   Range 0: [0, 1]
// CHECK: Allocations:
// CHECK:   Allocation 0: [0, 1] (alignment=2)

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @simple_range {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: no_ranges

// Test: no ranges - should still be satisfiable
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK-NOT: Range 0
// CHECK: Dependency Graph:

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @no_ranges {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: overlapping_ranges

// Test: overlapping ranges that share a common element
// Range A: [0, 1]
// Range B: [1, 2]
// Unsatisfiable: alignment=2 for both ranges creates conflicting constraints
// CHECK: Satisfiable: no
// CHECK: Ranges:
// CHECK-DAG:   Range {{[0-9]+}}: [0, 1]
// CHECK-DAG:   Range {{[0-9]+}}: [1, 2]

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @overlapping_ranges {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range_a = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v1, %v2 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: disjoint_ranges

// Test: disjoint ranges that don't share elements
// Range A: [0, 1]
// Range B: [2, 3]
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK-DAG:   Range {{[0-9]+}}: [0, 1]
// CHECK-DAG:   Range {{[0-9]+}}: [2, 3]
// CHECK: Allocations:
// CHECK-DAG:   Allocation {{[0-9]+}}: [0, 1] (alignment=2)
// CHECK-DAG:   Allocation {{[0-9]+}}: [2, 3] (alignment=2)

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @disjoint_ranges {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v3 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range_a = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v2, %v3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: subset_range

// Test: one range is a subset of another
// Range A: [0, 1, 2]
// Range B: [1]
// Unsatisfiable: alignment constraints conflict (Range A has alignment from size,
// Range B's element is at offset 1 in Range A which conflicts with Range B's alignment)
// CHECK: Satisfiable: no
// CHECK: Ranges:
// CHECK-DAG:   Range {{[0-9]+}}: [0, 1, 2]
// CHECK-DAG:   Range {{[0-9]+}}: [1]

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @subset_range {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range_a = make_register_range %v0, %v1, %v2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v1 : !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 3]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 1]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: cf_disjoint_branch_ranges

// Test: disjoint ranges in different branches (no shared elements)
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK-DAG:   Range {{[0-9]+}}: [0, 1]
// CHECK-DAG:   Range {{[0-9]+}}: [2, 3]

amdgcn.module @cf_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  amdgcn.kernel @cf_disjoint_branch_ranges {
    %cond = func.call @rand() : () -> i1
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v3 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %range_a = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    cf.br ^bb3
  ^bb2:
    %range_b = make_register_range %v2, %v3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    cf.br ^bb3
  ^bb3:
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: empty_kernel

// Test: empty kernel with no allocas or ranges
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK-NOT: Range 0

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @empty_kernel {
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: large_range

// Test: larger range with 4 elements
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK:   Range 0: [0, 1, 2, 3]
// CHECK: Allocations:
// CHECK:   Allocation 0: [0, 1, 2, 3] (alignment=4)

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @large_range {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v3 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range = make_register_range %v0, %v1, %v2, %v3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: same_range_twice

// Test: same range used multiple times (should be fine)
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK-DAG:   Range {{[0-9]+}}: [0, 1]

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @same_range_twice {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range_a = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: sgpr_range

// Test: SGPR range instead of VGPR
// CHECK: Satisfiable: yes
// CHECK: Ranges:
// CHECK:   Range 0: [0, 1]

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @sgpr_range {
    %0 = alloca : !amdgcn.sgpr
    %1 = alloca : !amdgcn.sgpr
    %v0 = test_inst outs %0 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %v1 = test_inst outs %1 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %range = make_register_range %v0, %v1 : !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %range : (!amdgcn.sgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Range Analysis Results ===
// CHECK-LABEL: Kernel: alignment_conflict

// Test: conflicting alignment requirements
// Range A: [0,1,2,3] has alignment 4 - element 1 is at offset 1
// Range B: [1,2] has alignment 2 - element 1 needs to be at offset 0 (mod 2)
// CHECK: Satisfiable: no

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @alignment_conflict {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v3 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %range_a = make_register_range %v0, %v1, %v2, %v3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v1, %v2 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr_range<[? + 4]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr_range<[? + 2]>) -> ()
    end_kernel
  }
}
// -----

// Test: phi coalescing scenario - the range analysis should be satisfiable.
// CHECK: Kernel: phi_coalescing_2
// CHECK: Satisfiable: yes
amdgcn.module @ra_phi_coalescing_2 target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_2 {
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    // While %7, %8 don't interfere in this block, they interfere with %9, %10
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %cond = lsir.cmpi i32 eq %3, %c0 : !amdgcn.sgpr, i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%8, %2) because they are dead.
    %9 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %alloc0 = alloca : !amdgcn.vgpr
    %bb1 = test_inst outs %alloc0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%bb1 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    // Nevertheless, we can reuse the allocation of (%7, %1) because they are dead.
    %10 = test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %alloc1 = alloca : !amdgcn.vgpr
    %bb2 = test_inst outs %alloc1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%bb2 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %val : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: phi coalescing scenario - the range analysis should be satisfiable.
// CHECK: Kernel: phi_coalescing_3
// CHECK: Satisfiable: yes
amdgcn.module @ra_phi_coalescing_3 target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_3 {
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    // While %7, %8 don't interfere in this block, they interfere with %9, %10
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %cond = lsir.cmpi i32 eq %3, %c0 : !amdgcn.sgpr, i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%7 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%8 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %val, %7, %8 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: phi coalescing scenario - the range analysis should be satisfiable.
// CHECK: Kernel: phi_coalescing_4
// CHECK: Satisfiable: yes
amdgcn.module @ra_phi_coalescing_4 target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_4 {
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    // While %7, %8 don't interfere in this block, they interfere with %9, %10
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %cond = lsir.cmpi i32 eq %3, %c0 : !amdgcn.sgpr, i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%7 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%8 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %val, %7, %8 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}
