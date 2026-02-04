// RUN: aster-opt %s --test-liveness-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: no_interference_mixed

// Simple test: no interference, values die after use.
// Note: outs operands are always live (true SSA), but eq classes are only live
// when the result is actually used.

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: ]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: ]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: ]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 2]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: ]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 2, 3]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 2]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.test_inst outs %[[v0]] ins %[[v2]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v3]], eqClasses: 3]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 2, 3]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v3]], eqClasses: 3]

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  kernel @no_interference_mixed {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = test_inst outs %0 ins %2 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %5 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: interference_mixed_all_live

// Test: values interfere because they are all live at the final use

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2, 3]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3, 4]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2, 3]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3, 4]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.test_inst outs %[[v0]] ins %[[v2]], %[[v4]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], eqClasses: 0, 1, 3, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4, 5]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]], %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v4]], %[[v5]], %[[v6]], %[[v7]], eqClasses: 0, 1, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], eqClasses: 0, 1, 3, 4, 5]

// All of %0, %1, %4, %5 must be live here (4 eq classes)
// CHECK: Operation: amdgcn.test_inst ins %[[v6]], %[[v7]], %[[v4]], %[[v5]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v4]], %[[v5]], %[[v6]], %[[v7]], eqClasses: 0, 1, 4, 5]

// CHECK: Operation: amdgcn.end_kernel

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  kernel @interference_mixed_all_live {
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


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: interference_mixed_with_reuse

// Test: values can be reused after they die

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2, 3]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3, 4]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2, 3]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3, 4]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.test_inst outs %[[v0]] ins %[[v2]], %[[v4]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 3, 5]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4, 5]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]], %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v6]], %[[v7]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 3, 5]

// Only %6 and %7 are live here
// CHECK: Operation: amdgcn.test_inst ins %[[v6]], %[[v7]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v6]], %[[v7]], eqClasses: 0, 1]

// CHECK: Operation: amdgcn.end_kernel

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  kernel @interference_mixed_with_reuse {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.sgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.vgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = test_inst outs %0 ins %2, %4 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3, %5 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %6, %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: interference_cf

// Test: control flow - values live across branches

// CHECK: Operation: %[[v0:[0-9]*]] = func.call @rand() : () -> i1
// Note: %0 is **not** a register.
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], eqClasses: 0]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], %[[v3]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2, 3]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2, 3, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], eqClasses: 0, 1, 2, 3, 4]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v4]], %[[v5]], %[[v6]], %[[v7]], eqClasses: 0, 1, 3, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2, 3, 4, 5]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.test_inst outs %[[v2]] ins %[[v4]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], %[[v7]], %[[v8]], eqClasses: 0, 1, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v4]], %[[v5]], %[[v6]], %[[v7]], eqClasses: 0, 1, 3, 4, 5]

// CHECK: Operation: cf.cond_br %[[v0]], ^bb1, ^bb2
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], %[[v7]], %[[v8]], eqClasses: 0, 1, 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], %[[v7]], %[[v8]], eqClasses: 0, 1, 4, 5]

// In bb1: outs %5 is always live, eq class 4 only becomes live if result %9 is used (it's not)
// CHECK: Operation: %[[v9:[0-9]*]] = amdgcn.test_inst outs %[[v5]] ins %[[v7]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], %[[v7]], eqClasses: 0, 5]

// CHECK: Operation: cf.br ^bb3
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]

// In bb2: outs %6 is always live, eq class 5 only becomes live if result %10 is used (it's not)
// CHECK: Operation: %[[v10:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v8]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], %[[v8]], eqClasses: 1, 4]

// CHECK: Operation: cf.br ^bb3
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]

// %5 and %6 must both be live here (from different branches)
// CHECK: Operation: amdgcn.test_inst ins %[[v5]], %[[v6]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v6]], eqClasses: 4, 5]

// CHECK: Operation: amdgcn.end_kernel

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1

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
  ^bb1:
    %9 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:
    %10 = test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:
    test_inst ins %5, %6 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: test_make_range_liveness

// Test: make_register_range keeps values live

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// Note: %6 alloca not live after because %7 result is not used
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// Note: outs %6 is live (true SSA), but eq class 3 is not because %7 isn't used
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]], %[[v5]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_make_range_liveness {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp1 = alloca : !amdgcn.vgpr
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range, %intermediate_0 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: test_make_range_liveness_1

// Test: make_register_range with simultaneous use of intermediate value

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]], %[[v5]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_make_range_liveness_1 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Note: Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    %tmp1 = alloca : !amdgcn.vgpr
    // Note: intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    // Note: Use at the same time as the range.
    test_inst ins %range, %intermediate_0 : (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: test_make_range_liveness_2

// Test: intermediate value used before make_register_range

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]

// Use before the range - %5 dies here
// CHECK: Operation: amdgcn.test_inst ins %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v8]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v8]], eqClasses: 0, 1]

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_make_range_liveness_2 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Note: Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    %tmp1 = alloca : !amdgcn.vgpr
    // Note: intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    // Note: Use before the range.
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: test_make_range_liveness_3

// Test: intermediate value used after make_register_range

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE  AFTER: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v1]], %[[v2]], eqClasses: 0, 1]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 0, 1]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v4]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], %[[v6]], eqClasses: 0, 1, 2]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], %[[v5]], eqClasses: 0, 1, 2]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]]
// CHECK: LIVE  AFTER: [values: %[[v5]], eqClasses: 2]
// CHECK: LIVE BEFORE: [values: %[[v5]], %[[v8]], eqClasses: 0, 1, 2]

// Use after the range - %5 dies here
// CHECK: Operation: amdgcn.test_inst ins %[[v5]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v5]], eqClasses: 2]

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_make_range_liveness_3 {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    // Note: Intermediate computation - these allocas must not reuse %0 or %1's registers
    // because %2 and %3 are still live (used by make_register_range below),
    // which itself is used by range.

    %tmp1 = alloca : !amdgcn.vgpr
    // Note: intermediate_0 is used, it must not be clobbered
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp2 = alloca : !amdgcn.vgpr
    %intermediate_1 = test_inst outs %tmp2 ins %intermediate_0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    // Note: Use after the range.
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----


// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: reg_interference

// Test: reg_interference marks values as live

// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], eqClasses: 0]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]
// CHECK: LIVE BEFORE: [values: %[[v0]], eqClasses: 0]

// CHECK: Operation: amdgcn.test_inst ins %[[v0]], %[[v1]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v0]], %[[v1]], eqClasses: 0, 1]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], eqClasses: 2]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v2]], %[[v3]], eqClasses: 2, 3]
// CHECK: LIVE BEFORE: [values: %[[v2]], eqClasses: 2]

// CHECK: Operation: amdgcn.test_inst ins %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v2]], %[[v3]], eqClasses: 2, 3]

// CHECK: Operation: amdgcn.reg_interference %[[v0]], %[[v2]], %[[v3]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v4]], eqClasses: 4]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v4]], %[[v5]], eqClasses: 4, 5]
// CHECK: LIVE BEFORE: [values: %[[v4]], eqClasses: 4]

// CHECK: Operation: amdgcn.test_inst ins %[[v4]], %[[v5]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v4]], %[[v5]], eqClasses: 4, 5]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v6]], eqClasses: 6]
// CHECK: LIVE BEFORE: []

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE  AFTER: [values: %[[v6]], %[[v7]], eqClasses: 6, 7]
// CHECK: LIVE BEFORE: [values: %[[v6]], eqClasses: 6]

// CHECK: Operation: amdgcn.test_inst ins %[[v6]], %[[v7]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: [values: %[[v6]], %[[v7]], eqClasses: 6, 7]

// CHECK: Operation: amdgcn.reg_interference %[[v4]], %[[v1]], %[[v3]], %[[v7]]
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE  AFTER: []
// CHECK: LIVE BEFORE: []

// CHECK: === End Analysis Results ===

amdgcn.module @liveness_tests target = <gfx942> isa = <cdna3> {
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
// -----

// Test: phi coalescing - allocas in different branches that flow to the same block
// argument get phi-coalesced into the same equivalence class (eq class 6).
// CHECK: Kernel: phi_coalescing_2
// CHECK: Equivalence Classes:
// CHECK-DAG:   EqClass 0: [%0, %6]
// CHECK-DAG:   EqClass 1: [%1, %7]
// CHECK-DAG:   EqClass 2: [%2]
// CHECK-DAG:   EqClass 3: [%3]
// CHECK-DAG:   EqClass 4: [%4, %9]
// CHECK-DAG:   EqClass 5: [%5, %12]
// Phi-coalesced allocas %10 and %13 share eq class 6
// CHECK-DAG:   EqClass 6: [%10, %11, %13, %14]
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

// Test: phi coalescing - values %7 and %8 flow to same block arg, so their source
// allocas %1 and %2 (now %0 and %1) get phi-coalesced into eq class 0.
// CHECK: Kernel: phi_coalescing_3
// CHECK: Equivalence Classes:
// Allocas %0 and %1 are phi-coalesced because %6 and %7 flow to same block arg
// CHECK-DAG:   EqClass 0: [%0, %1, %6, %7]
// CHECK-DAG:   EqClass 1: [%2]
// CHECK-DAG:   EqClass 2: [%3]
// CHECK-DAG:   EqClass 3: [%4]
// CHECK-DAG:   EqClass 4: [%5]
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

// Test: same as phi_coalescing_3 - phi coalescing reduces eq classes.
// CHECK: Kernel: phi_coalescing_4
// CHECK: Equivalence Classes:
// CHECK-DAG:   EqClass 0: [%0, %1, %6, %7]
// CHECK-DAG:   EqClass 1: [%2]
// CHECK-DAG:   EqClass 2: [%3]
// CHECK-DAG:   EqClass 3: [%4]
// CHECK-DAG:   EqClass 4: [%5]
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
