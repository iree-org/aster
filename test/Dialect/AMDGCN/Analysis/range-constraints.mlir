// RUN: aster-opt %s --split-input-file --test-amdgcn-range-analysis --verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Symbol: simple_range
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 2, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @simple_range {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    %4 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %4 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Symbol: no_ranges
// CHECK:  No range constraints
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @no_ranges {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    test_inst ins %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [5 = `%{{.*}}`]
// CHECK:  Symbol: disjoint_ranges
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 2, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`]>
// CHECK:    Constraint 1: range_constraint<alignment = 2, allocations = [2 = `%{{.*}}`, 3 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @disjoint_ranges {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %2 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %3 : (!amdgcn.vgpr<?>) -> ()
    %8 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %9 = make_register_range %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %8 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    test_inst ins %9 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}} : !amdgcn.vgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Symbol: subset_range
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 4, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @subset_range {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %2 : (!amdgcn.vgpr<?>) -> ()
    %6 = make_register_range %0, %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %7 = make_register_range %1 : !amdgcn.vgpr<?>
    test_inst ins %6 : (!amdgcn.vgpr<[? : ? + 3]>) -> ()
    test_inst ins %7 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = func.call @rand() : () -> i1`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [6 = `%{{.*}}`]
// CHECK:  Symbol: rand
// CHECK:  No range constraints
// CHECK:  Symbol: cf_disjoint_branch_ranges
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 2, allocations = [1 = `%{{.*}}`, 2 = `%{{.*}}`]>
// CHECK:    Constraint 1: range_constraint<alignment = 2, allocations = [3 = `%{{.*}}`, 4 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @cf_disjoint_branch_ranges {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    %4 = alloca : !amdgcn.vgpr<?>
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %2 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %3 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %4 : (!amdgcn.vgpr<?>) -> ()
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // CHECK:  pred: ^bb0
    %9 = make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %9 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    cf.br ^bb3
  ^bb2:  // CHECK:  pred: ^bb0
    %10 = make_register_range %3, %4 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %10 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    cf.br ^bb3
  ^bb3:  // CHECK:  2 preds: ^bb1, ^bb2
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Symbol: empty_kernel
// CHECK:  No range constraints
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @empty_kernel {
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Symbol: large_range
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 4, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`, 3 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @large_range {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %2 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %3 : (!amdgcn.vgpr<?>) -> ()
    %8 = make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %8 : (!amdgcn.vgpr<[? : ? + 4]>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Symbol: same_range_twice
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 2, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @same_range_twice {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.vgpr<?>) -> ()
    %4 = make_register_range %0, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %4 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    test_inst ins %4 : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.make_register_range %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Symbol: sgpr_range
// CHECK:  Range constraints:
// CHECK:    Constraint 0: range_constraint<alignment = 2, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @sgpr_range {
    %0 = alloca : !amdgcn.sgpr<?>
    %1 = alloca : !amdgcn.sgpr<?>
    test_inst outs %0 : (!amdgcn.sgpr<?>) -> ()
    test_inst outs %1 : (!amdgcn.sgpr<?>) -> ()
    %4 = make_register_range %0, %1 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
    test_inst ins %4 : (!amdgcn.sgpr<[? : ? + 2]>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = arith.constant 0 : i32`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [6 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32`
// CHECK:  results: [7 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [8 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [9 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [10 = `%{{.*}}`]
// CHECK:  Symbol: phi_coalescing_2
// CHECK:  No range constraints
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_2 {
    %c0_i32 = arith.constant 0 : i32
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.sgpr<?>
    %3 = alloca : !amdgcn.sgpr<?>
    %4 = alloca : !amdgcn.vgpr<?>
    %5 = alloca : !amdgcn.vgpr<?>
    test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
    test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
    %8 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
    %9 = alloca : !amdgcn.vgpr<?>
    cf.cond_br %8, ^bb1, ^bb2
  ^bb1:  // CHECK:  pred: ^bb0
    test_inst outs %4 ins %0 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %11 = alloca : !amdgcn.vgpr<?>
    test_inst outs %11 : (!amdgcn.vgpr<?>) -> ()
    lsir.copy %9, %11 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb2:  // CHECK:  pred: ^bb0
    test_inst outs %5 ins %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %15 = alloca : !amdgcn.vgpr<?>
    test_inst outs %15 : (!amdgcn.vgpr<?>) -> ()
    lsir.copy %9, %15 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb3:  // CHECK:  2 preds: ^bb1, ^bb2
    test_inst ins %9 : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----
// CHECK-LABEL:  SSA map
// CHECK:  Operation: `%{{.*}} = arith.constant 0 : i32`
// CHECK:  results: [0 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [1 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [2 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [3 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.sgpr<?>`
// CHECK:  results: [4 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = lsir.cmpi i32 eq %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, i32`
// CHECK:  results: [5 = `%{{.*}}`]
// CHECK:  Operation: `%{{.*}} = amdgcn.alloca : !amdgcn.vgpr<?>`
// CHECK:  results: [6 = `%{{.*}}`]
// CHECK:  Symbol: phi_coalescing_3
// CHECK:  No range constraints
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @phi_coalescing_3 {
    %c0_i32 = arith.constant 0 : i32
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.sgpr<?>
    %3 = alloca : !amdgcn.sgpr<?>
    test_inst outs %0 ins %2 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
    test_inst outs %1 ins %3 : (!amdgcn.vgpr<?>, !amdgcn.sgpr<?>) -> ()
    %6 = lsir.cmpi i32 eq %2, %c0_i32 : !amdgcn.sgpr<?>, i32
    %7 = alloca : !amdgcn.vgpr<?>
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // CHECK:  pred: ^bb0
    lsir.copy %7, %0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb2:  // CHECK:  pred: ^bb0
    lsir.copy %7, %1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^bb3
  ^bb3:  // CHECK:  2 preds: ^bb1, ^bb2
    test_inst ins %7, %0, %1 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  // expected-error@+1 {{Failed to run range constraint analysis}}
  amdgcn.kernel @overlapping_ranges {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %range_a = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v1, %v2 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr<[? + 2]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  // expected-error@+1 {{Failed to run range constraint analysis}}
  amdgcn.kernel @alignment_conflict {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %0 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %v1 = test_inst outs %1 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %v2 = test_inst outs %2 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %v3 = test_inst outs %3 : (!amdgcn.vgpr) -> (!amdgcn.vgpr)
    %range_a = make_register_range %v0, %v1, %v2, %v3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    %range_b = make_register_range %v1, %v2 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range_a : (!amdgcn.vgpr<[? + 4]>) -> ()
    test_inst ins %range_b : (!amdgcn.vgpr<[? + 2]>) -> ()
    end_kernel
  }
}


// -----

//CHECK-LABEL: Symbol: constraintrs_descending_order
//CHECK: Range constraints:
//CHECK:   Constraint 0: range_constraint<alignment = 4, allocations = [0 = `%{{.*}}`, 1 = `%{{.*}}`, 2 = `%{{.*}}`, 3 = `%{{.*}}`]>
//CHECK:   Constraint 1: range_constraint<alignment = 2, allocations = [4 = `%{{.*}}`, 5 = `%{{.*}}`]>
amdgcn.module @range_tests target = <gfx942> isa = <cdna3> {
  kernel @constraintrs_descending_order {
    %0 = alloca : !amdgcn.vgpr<?>
    %1 = alloca : !amdgcn.vgpr<?>
    %2 = alloca : !amdgcn.vgpr<?>
    %3 = alloca : !amdgcn.vgpr<?>
    %4 = alloca : !amdgcn.vgpr<?>
    %5 = alloca : !amdgcn.vgpr<?>
    %6 = make_register_range %4, %5 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %7 = make_register_range %0, %1, %2, %3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    test_inst ins %6, %7 : (!amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 4]>) -> ()
    end_kernel
  }
}
