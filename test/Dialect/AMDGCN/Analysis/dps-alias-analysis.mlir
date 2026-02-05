// RUN: aster-opt %s --test-dps-alias-analysis --split-input-file 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Trivial cases
//===----------------------------------------------------------------------===//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: empty_kernel

// Test: empty kernel with no allocas
// CHECK: amdgcn.kernel @empty_kernel {
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK-NOT: EqClass
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @empty_kernel {
    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: single_alloca_no_use

// Test: single alloca with no uses
// CHECK: amdgcn.kernel @single_alloca_no_use {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @single_alloca_no_use {
    %0 = alloca : !amdgcn.vgpr
    end_kernel
  }
}

// -----

//===----------------------------------------------------------------------===//
// Basic straight-line code
//===----------------------------------------------------------------------===//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: simple_dps

// Test: simple DPS - alloca produces result, result goes to equivalence class
// CHECK: amdgcn.kernel @simple_dps {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v3:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]*]] = test_inst outs %[[v0]] ins %[[v2]]
// CHECK:   %[[v5:[0-9]*]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v4]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v5]]]
// CHECK:   EqClass 2: [%[[v2]]]
// CHECK:   EqClass 3: [%[[v3]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @simple_dps {
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

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: deep_dps_chain

// Test: chain of DPS operations - each result used as input to next
// CHECK: amdgcn.kernel @deep_dps_chain {
// CHECK:   %[[s0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[s1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[s2:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[s3:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v0:[0-9]*]] = test_inst outs %[[s0]]
// CHECK:   %[[v1:[0-9]*]] = test_inst outs %[[s1]] ins %[[v0]]
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[s2]] ins %[[v1]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[s3]] ins %[[v2]]
// CHECK:   test_inst ins %[[v3]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[s0]], %[[v0]]]
// CHECK:   EqClass 1: [%[[s1]], %[[v1]]]
// CHECK:   EqClass 2: [%[[s2]], %[[v2]]]
// CHECK:   EqClass 3: [%[[s3]], %[[v3]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @deep_dps_chain {
    %s0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.vgpr
    %s2 = alloca : !amdgcn.vgpr
    %s3 = alloca : !amdgcn.vgpr
    %v0 = test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %s1 ins %v0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %s2 ins %v1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %v3 = test_inst outs %s3 ins %v2 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %v3 : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: many_allocas

// Test: many allocas with simple DPS chains
// CHECK: amdgcn.kernel @many_allocas {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v3:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[r0:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[r1:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[r2:[0-9]*]] = test_inst outs %[[v2]]
// CHECK:   %[[r3:[0-9]*]] = test_inst outs %[[v3]]
// CHECK:   %[[r4:[0-9]*]] = test_inst outs %[[v4]]
// CHECK:   test_inst ins %[[r0]], %[[r1]], %[[r2]], %[[r3]], %[[r4]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[r0]]]
// CHECK:   EqClass 1: [%[[v1]], %[[r1]]]
// CHECK:   EqClass 2: [%[[v2]], %[[r2]]]
// CHECK:   EqClass 3: [%[[v3]], %[[r3]]]
// CHECK:   EqClass 4: [%[[v4]], %[[r4]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @many_allocas {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.vgpr
    %4 = alloca : !amdgcn.vgpr
    %r0 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %r1 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %r2 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %r3 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %r4 = test_inst outs %4 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %r0, %r1, %r2, %r3, %r4 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

//===----------------------------------------------------------------------===//
// Mixed register types (SGPR + VGPR)
//===----------------------------------------------------------------------===//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: sgpr_and_vgpr_mixed

// Test: mixed SGPR and VGPR allocas with interactions
// CHECK: amdgcn.kernel @sgpr_and_vgpr_mixed {
// CHECK:   %[[s0:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[s1:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[rs0:[0-9]*]] = test_inst outs %[[s0]]
// CHECK:   %[[rv0:[0-9]*]] = test_inst outs %[[v0]] ins %[[rs0]]
// CHECK:   %[[rs1:[0-9]*]] = test_inst outs %[[s1]]
// CHECK:   %[[rv1:[0-9]*]] = test_inst outs %[[v1]] ins %[[rs1]], %[[rv0]]
// CHECK:   test_inst ins %[[rv1]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[s0]], %[[rs0]]]
// CHECK:   EqClass 1: [%[[v0]], %[[rv0]]]
// CHECK:   EqClass 2: [%[[s1]], %[[rs1]]]
// CHECK:   EqClass 3: [%[[v1]], %[[rv1]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @sgpr_and_vgpr_mixed {
    %s0 = alloca : !amdgcn.sgpr
    %v0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.sgpr
    %v1 = alloca : !amdgcn.vgpr
    %rs0 = test_inst outs %s0 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %rv0 = test_inst outs %v0 ins %rs0 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %rs1 = test_inst outs %s1 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %rv1 = test_inst outs %v1 ins %rs1, %rv0 : (!amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %rv1 : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: diamond_no_merge

// Test: diamond control flow with independent branch operations
// CHECK: amdgcn.kernel @diamond_no_merge {
// CHECK:   %[[cond:[0-9]*]] = func.call @rand() : () -> i1
// CHECK:   %[[s0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[s1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   cf.cond_br %[[cond]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[a:[0-9]*]] = test_inst outs %[[s0]]
// CHECK:   test_inst ins %[[a]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   %[[b:[0-9]*]] = test_inst outs %[[s1]]
// CHECK:   test_inst ins %[[b]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[s0]], %[[a]]]
// CHECK:   EqClass 1: [%[[s1]], %[[b]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  amdgcn.kernel @diamond_no_merge {
    %cond = func.call @rand() : () -> i1
    %s0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %a = test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %a : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb2:
    %b = test_inst outs %s1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %b : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb3:
    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: diamond_values_live_across

// Test: diamond control flow with values live across branches
// CHECK: amdgcn.kernel @diamond_values_live_across {
// CHECK:   %[[v0:[0-9]*]] = func.call @rand() : () -> i1
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v3:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v5:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v6:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v7:[0-9]*]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   %[[v8:[0-9]*]] = test_inst outs %[[v2]] ins %[[v4]]
// CHECK:   cf.cond_br %[[v0]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[v9:[0-9]*]] = test_inst outs %[[v5]] ins %[[v7]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   %[[v10:[0-9]*]] = test_inst outs %[[v6]] ins %[[v8]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   test_inst ins %[[v5]], %[[v6]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v1]], %[[v7]]]
// CHECK:   EqClass 1: [%[[v2]], %[[v8]]]
// CHECK:   EqClass 2: [%[[v3]]]
// CHECK:   EqClass 3: [%[[v4]]]
// CHECK:   EqClass 4: [%[[v5]], %[[v9]]]
// CHECK:   EqClass 5: [%[[v6]], %[[v10]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1

  amdgcn.kernel @diamond_values_live_across {
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

//===----------------------------------------------------------------------===//
// Register ranges (make_register_range)
//===----------------------------------------------------------------------===//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: make_range_basic

// Test: make_register_range groups values into same equivalence class
// CHECK: amdgcn.kernel @make_range_basic {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[v4:[0-9]*]] = make_register_range %[[v2]], %[[v3]]
// CHECK:   test_inst ins %[[v4]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v2]], %[[v4]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v3]], %[[v4]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @make_range_basic {
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

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: make_range_with_intermediate

// Test: make_register_range with intermediate computations
// CHECK: amdgcn.kernel @make_range_with_intermediate {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]*]] = test_inst outs %[[v4]] ins %[[v3]]
// CHECK:   %[[v6:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v7:[0-9]*]] = test_inst outs %[[v6]] ins %[[v5]]
// CHECK:   %[[v8:[0-9]*]] = make_register_range %[[v2]], %[[v3]]
// CHECK:   test_inst ins %[[v8]], %[[v5]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v2]], %[[v8]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v3]], %[[v8]]]
// CHECK:   EqClass 2: [%[[v4]], %[[v5]]]
// CHECK:   EqClass 3: [%[[v6]], %[[v7]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @make_range_with_intermediate {
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

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: make_range_use_before

// Test: intermediate value used before make_register_range
// CHECK: amdgcn.kernel @make_range_use_before {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]*]] = test_inst outs %[[v4]] ins %[[v3]]
// CHECK:   test_inst ins %[[v5]]
// CHECK:   %[[v6:[0-9]*]] = make_register_range %[[v2]], %[[v3]]
// CHECK:   test_inst ins %[[v6]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v2]], %[[v6]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v3]], %[[v6]]]
// CHECK:   EqClass 2: [%[[v4]], %[[v5]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @make_range_use_before {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp1 = alloca : !amdgcn.vgpr
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    // Use before the range
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: make_range_use_after

// Test: intermediate value used after make_register_range
// CHECK: amdgcn.kernel @make_range_use_after {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]*]] = test_inst outs %[[v4]] ins %[[v3]]
// CHECK:   %[[v6:[0-9]*]] = make_register_range %[[v2]], %[[v3]]
// CHECK:   test_inst ins %[[v6]]
// CHECK:   test_inst ins %[[v5]]
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v2]], %[[v6]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v3]], %[[v6]]]
// CHECK:   EqClass 2: [%[[v4]], %[[v5]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @make_range_use_after {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr

    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    %tmp1 = alloca : !amdgcn.vgpr
    %intermediate_0 = test_inst outs %tmp1 ins %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
    test_inst ins %range : (!amdgcn.vgpr_range<[? + 2]>) -> ()

    // Use after the range
    test_inst ins %intermediate_0 : (!amdgcn.vgpr) -> ()

    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: phi_coalescing_2

// Test: phi coalescing - allocas flowing to the same block argument get the same
// equivalence class due to ValueProvenanceAnalysis.
// CHECK: amdgcn.kernel @phi_coalescing_2 {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v3:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v6:[0-9]*]] = test_inst outs %[[v0]] ins %[[v2]]
// CHECK:   %[[v7:[0-9]*]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   lsir.cmpi i32 eq
// CHECK:   cf.cond_br
// CHECK: ^bb1:
// CHECK:   %[[v9:[0-9]*]] = test_inst outs %[[v4]] ins %[[v6]]
// CHECK:   %[[v10:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v11:[0-9]*]] = test_inst outs %[[v10]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   %[[v12:[0-9]*]] = test_inst outs %[[v5]] ins %[[v7]]
// CHECK:   %[[v13:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v14:[0-9]*]] = test_inst outs %[[v13]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb3
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK:   EqClass 0: [%[[v0]], %[[v6]]]
// CHECK:   EqClass 1: [%[[v1]], %[[v7]]]
// CHECK:   EqClass 2: [%[[v2]]]
// CHECK:   EqClass 3: [%[[v3]]]
// CHECK:   EqClass 4: [%[[v4]], %[[v9]]]
// CHECK:   EqClass 5: [%[[v5]], %[[v12]]]
// Allocas %10 and %13 are phi-coalesced because they flow to the same block argument
// CHECK:   EqClass 6: [%[[v10]], %[[v11]], %[[v13]], %[[v14]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @phi_coalescing_2 {
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %cond = lsir.cmpi i32 eq %3, %c0 : !amdgcn.sgpr, i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %9 = test_inst outs %5 ins %7 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %alloc0 = alloca : !amdgcn.vgpr
    %bb1 = test_inst outs %alloc0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%bb1 : !amdgcn.vgpr)
  ^bb2:
    %10 = test_inst outs %6 ins %8 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %alloc1 = alloca : !amdgcn.vgpr
    %bb2 = test_inst outs %alloc1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%bb2 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):
    test_inst ins %val : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: phi_coalescing_3

// Test: phi coalescing when values from different allocas flow to same block argument.
// %7 and %8 both flow to ^bb3's block argument, so their source allocas %1 and %2
// must be in the same equivalence class.
// CHECK: amdgcn.kernel @phi_coalescing_3 {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v3:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v6:[0-9]*]] = test_inst outs %[[v0]] ins %[[v2]]
// CHECK:   %[[v7:[0-9]*]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   lsir.cmpi i32 eq
// CHECK:   cf.cond_br
// CHECK: ^bb1:
// CHECK:   cf.br ^bb3(%[[v6]] : !amdgcn.vgpr)
// CHECK: ^bb2:
// CHECK:   cf.br ^bb3(%[[v7]] : !amdgcn.vgpr)
// CHECK: ^bb3
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// Allocas %0 and %1 are phi-coalesced because their results %6 and %7 flow to same block arg
// CHECK:   EqClass 0: [%[[v0]], %[[v1]], %[[v6]], %[[v7]]]
// CHECK:   EqClass 1: [%[[v2]]]
// CHECK:   EqClass 2: [%[[v3]]]
// CHECK:   EqClass 3: [%[[v4]]]
// CHECK:   EqClass 4: [%[[v5]]]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @phi_coalescing_3 {
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %3 = alloca : !amdgcn.sgpr
    %4 = alloca : !amdgcn.sgpr
    %5 = alloca : !amdgcn.vgpr
    %6 = alloca : !amdgcn.vgpr
    %7 = test_inst outs %1 ins %3 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %8 = test_inst outs %2 ins %4 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %cond = lsir.cmpi i32 eq %3, %c0 : !amdgcn.sgpr, i32
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%7 : !amdgcn.vgpr)
  ^bb2:
    cf.br ^bb3(%8 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):
    test_inst ins %val, %7, %8 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: split_register_range_normal_match

// Test: Normal case where split_register_range receives correct eqClassId count
// This verifies the baseline happy path still works correctly
// CHECK: amdgcn.kernel @split_register_range_normal_match {
// CHECK:   %[[a0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[a1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v0:[0-9]*]] = test_inst outs %[[a0]]
// CHECK:   %[[v1:[0-9]*]] = test_inst outs %[[a1]]
// CHECK:   %[[range:[0-9]*]] = make_register_range %[[v0]], %[[v1]]
// CHECK:   %[[r2:[0-9]*]]:2 = split_register_range %[[range]]
// CHECK:   test_inst ins %[[r2]]#0, %[[r2]]#1
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: no
// CHECK: Equivalence Classes:
// CHECK-DAG: EqClass {{[0-9]+}}: [%[[a0]], %[[v0]], %[[range]], %[[r2]]#0, %[[r2]]#1]
// CHECK-DAG: EqClass {{[0-9]+}}: [%[[a1]], %[[v1]], %[[range]], %[[r2]]#0, %[[r2]]#1]
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @split_register_range_normal_match {
    %a0 = alloca : !amdgcn.vgpr
    %a1 = alloca : !amdgcn.vgpr

    %v0 = test_inst outs %a0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %v1 = test_inst outs %a1 : (!amdgcn.vgpr) -> !amdgcn.vgpr

    %range = make_register_range %v0, %v1 : !amdgcn.vgpr, !amdgcn.vgpr
    %r0, %r1 = split_register_range %range : !amdgcn.vgpr_range<[? + 2]>

    test_inst ins %r0, %r1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: split_register_range on register_range from loop iter_args
// Reproduces the issue where split_register_range receives a register_range as a block
// argument from control flow (iter_args), and must handle potentially mismatched alias info.
//

// CHECK: === DPS Alias Analysis Results ===
// CHECK-LABEL: Kernel: split_register_range_with_iter_args
// CHECK:   %[[c0:[a-zA-Z_0-9]*]] = arith.constant 0 : i32
// CHECK:   %[[c1:[a-zA-Z_0-9]*]] = arith.constant 1 : i32
// CHECK:   %[[c2:[a-zA-Z_0-9]*]] = arith.constant 2 : i32
// CHECK:   %[[alloc_cc:[a-zA-Z_0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[alloc_cd:[a-zA-Z_0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[alloc_ce:[a-zA-Z_0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[alloc_cf:[a-zA-Z_0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[alloc_d0:[a-zA-Z_0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   cf.br
// CHECK: ^bb1
// CHECK:   make_register_range
// CHECK:   amdgcn.vop3p.vop3p_mai
// CHECK:   cf.cond_br
// CHECK: ^bb2
// CHECK:   split_register_range
// CHECK:   end_kernel
// CHECK: }
// CHECK: Ill-formed IR: yes
// CHECK: Equivalence Classes:
// CHECK: === End Analysis Results ===

amdgcn.module @dps_alias_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @split_register_range_with_iter_args {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    // Allocas for initial register range
    %alloc0 = alloca : !amdgcn.vgpr
    %alloc1 = alloca : !amdgcn.vgpr
    %alloc2 = alloca : !amdgcn.vgpr
    %alloc3 = alloca : !amdgcn.vgpr
    %alloc_sgpr = alloca : !amdgcn.sgpr

    // Allocas for loop body
    %alloc_body0 = alloca : !amdgcn.vgpr
    %alloc_body1 = alloca : !amdgcn.vgpr
    %alloc_body2 = alloca : !amdgcn.vgpr
    %alloc_body3 = alloca : !amdgcn.vgpr

    // Create initial register range and mock A/B
    %init0 = test_inst outs %alloc0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %init1 = test_inst outs %alloc1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %init2 = test_inst outs %alloc2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %init3 = test_inst outs %alloc3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %reg_range_init = make_register_range %init0, %init1, %init2, %init3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    %sgpr_init = sop1 s_mov_b32 outs %alloc_sgpr ins %c0_i32 : !amdgcn.sgpr, i32

    %mock_a0 = test_inst outs %alloc_body0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %mock_a1 = test_inst outs %alloc_body1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %mock_a = make_register_range %mock_a0, %mock_a1 : !amdgcn.vgpr, !amdgcn.vgpr

    // Enter loop with iter_args
    cf.br ^bb1(%sgpr_init, %reg_range_init : !amdgcn.sgpr, !amdgcn.vgpr_range<[? + 4]>)

  ^bb1(%sgpr_iter: !amdgcn.sgpr, %reg_iter: !amdgcn.vgpr_range<[? + 4]>):
    // Create new register range for MFMA in loop body
    %body_init0 = test_inst outs %alloc_body2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %body_init1 = test_inst outs %alloc_body3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %body_reg = make_register_range %body_init0, %body_init1, %alloc_body2, %alloc_body3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // Use register_range from iter_args as accumulator to MFMA
    %mfma_result = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %body_reg, %mock_a, %mock_a, %reg_iter : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>

    // Loop increment
    %sgpr_next = sop2 s_add_u32 outs %alloc_sgpr ins %sgpr_iter, %c1_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
    %loop_cond = lsir.cmpi i32 slt %sgpr_next, %c2_i32 : !amdgcn.sgpr, i32
    cf.cond_br %loop_cond, ^bb1(%sgpr_next, %mfma_result : !amdgcn.sgpr, !amdgcn.vgpr_range<[? + 4]>), ^bb2

  ^bb2:
    // split_register_range on MFMA result (which came from iter_args)
    %split:4 = split_register_range %mfma_result : !amdgcn.vgpr_range<[? + 4]>
    end_kernel
  }
}
