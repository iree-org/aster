// RUN: aster-opt %s --test-liveness-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK: === Liveness Analysis Results ===
// CHECK-LABEL: Kernel: no_interference_mixed

// Simple test: no interference, values die after use
// CHECK: Operation: %[[v0:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v2]]]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: [%[[v2]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

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
// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr

// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]], %[[v4]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]], %[[v4]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]], %[[v4]], %[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.test_inst outs %[[v0]] ins %[[v2]], %[[v4]]
// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]], %[[v4]], %[[v5]]]
// CHECK: LIVE AFTER: [%[[v3]], %[[v4]], %[[v5]], %[[v6]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]], %[[v5]]
// CHECK: LIVE BEFORE: [%[[v3]], %[[v4]], %[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: [%[[v4]], %[[v5]], %[[v6]], %[[v7]]]

// All of %0, %1, %4, %5 must be live here (4 eq classes)
// CHECK: Operation: amdgcn.test_inst ins %[[v6]], %[[v7]], %[[v4]], %[[v5]]
// CHECK: LIVE BEFORE: [%[[v4]], %[[v5]], %[[v6]], %[[v7]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE BEFORE: []

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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v2]]]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: [%[[v2]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]], %[[v4]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]], %[[v4]]]
// CHECK: LIVE AFTER: [%[[v2]], %[[v3]], %[[v4]], %[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.test_inst outs %[[v0]] ins %[[v2]], %[[v4]]
// CHECK: LIVE BEFORE: [%[[v2]], %[[v3]], %[[v4]], %[[v5]]]
// CHECK: LIVE AFTER: [%[[v3]], %[[v5]], %[[v6]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]], %[[v5]]
// CHECK: LIVE BEFORE: [%[[v3]], %[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: [%[[v6]], %[[v7]]]

// Only %6 and %7 are live here
// CHECK: Operation: amdgcn.test_inst ins %[[v6]], %[[v7]]
// CHECK: LIVE BEFORE: [%[[v6]], %[[v7]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE BEFORE: []

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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v0]]]

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v0]]]
// CHECK: LIVE AFTER: [%[[v0]]]

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v0]]]
// CHECK: LIVE AFTER: [%[[v0]]]

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: [%[[v0]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK: LIVE BEFORE: [%[[v0]], %[[v3]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v3]], %[[v4]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v0]], %[[v3]], %[[v4]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v3]], %[[v4]], %[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v0]], %[[v3]], %[[v4]], %[[v5]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v3]], %[[v4]], %[[v5]], %[[v6]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v1]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v0]], %[[v3]], %[[v4]], %[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v4]], %[[v5]], %[[v6]], %[[v7]]]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.test_inst outs %[[v2]] ins %[[v4]]
// CHECK: LIVE BEFORE: [%[[v0]], %[[v4]], %[[v5]], %[[v6]], %[[v7]]]
// CHECK: LIVE AFTER: [%[[v0]], %[[v5]], %[[v6]], %[[v7]], %[[v8]]]

// CHECK: Operation: cf.cond_br %[[v0]], ^bb1, ^bb2
// CHECK: LIVE BEFORE: [%[[v0]], %[[v5]], %[[v6]], %[[v7]], %[[v8]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v6]], %[[v7]], %[[v8]]]

// In bb1: %6 and %7 live (for final use)
// CHECK: Operation: %[[v9:[0-9]*]] = amdgcn.test_inst outs %[[v5]] ins %[[v7]]
// CHECK: LIVE BEFORE: [%[[v6]], %[[v7]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v6]]]

// CHECK: Operation: cf.br ^bb3
// CHECK: LIVE BEFORE: [%[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v6]]]

// In bb2: %5 and %8 live (for final use)
// CHECK: Operation: %[[v10:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v8]]
// CHECK: LIVE BEFORE: [%[[v5]], %[[v8]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v6]]]

// CHECK: Operation: cf.br ^bb3
// CHECK: LIVE BEFORE: [%[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v6]]]

// %5 and %6 must both be live here (from different branches)
// CHECK: Operation: amdgcn.test_inst ins %[[v5]], %[[v6]]
// CHECK: LIVE BEFORE: [%[[v5]], %[[v6]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
// CHECK: LIVE BEFORE: []

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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v8]]]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]], %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]], %[[v8]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v8]]]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]], %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]], %[[v8]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// Use before the range - %5 dies here
// CHECK: Operation: amdgcn.test_inst ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v8]]]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]]
// CHECK: LIVE BEFORE: [%[[v8]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
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
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v1:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v2:[0-9]*]] = amdgcn.test_inst outs %[[v0]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: []

// CHECK: Operation: %[[v3:[0-9]*]] = amdgcn.test_inst outs %[[v1]]
// CHECK: LIVE BEFORE: []
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v4:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v3]]]

// CHECK: Operation: %[[v5:[0-9]*]] = amdgcn.test_inst outs %[[v4]] ins %[[v3]]
// CHECK: LIVE BEFORE: [%[[v3]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v6:[0-9]*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v7:[0-9]*]] = amdgcn.test_inst outs %[[v6]] ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// CHECK: Operation: %[[v8:[0-9]*]] = amdgcn.make_register_range %[[v2]], %[[v3]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: [%[[v5]], %[[v8]]]

// CHECK: Operation: amdgcn.test_inst ins %[[v8]]
// CHECK: LIVE BEFORE: [%[[v5]], %[[v8]]]
// CHECK: LIVE AFTER: [%[[v5]]]

// Use after the range - %5 dies here
// CHECK: Operation: amdgcn.test_inst ins %[[v5]]
// CHECK: LIVE BEFORE: [%[[v5]]]
// CHECK: LIVE AFTER: []

// CHECK: Operation: amdgcn.end_kernel
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
