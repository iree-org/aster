// RUN: aster-opt %s --test-value-provenance-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: basic_dps_chain

// Test: Basic chain - alloca flows through test_inst result
// CHECK: amdgcn.kernel @basic_dps_chain {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]] ins %[[v1]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]
// CHECK:   %[[v1]] -> %[[v1]]
// CHECK:   %[[v2]] -> %[[v0]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @basic_dps_chain {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.sgpr
    %2 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: chained_dps

// Test: Chained operations - provenance flows through chain
// CHECK: amdgcn.kernel @chained_dps {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v2]] ins %[[v1]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]
// CHECK:   %[[v1]] -> %[[v1]]
// CHECK:   %[[v2]] -> %[[v0]]
// CHECK:   %[[v3]] -> %[[v0]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @chained_dps {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.sgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %2 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: multiple_allocas

// Test: Multiple independent allocas
// CHECK: amdgcn.kernel @multiple_allocas {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]
// CHECK:   %[[v1]] -> %[[v1]]
// CHECK:   %[[v2]] -> %[[v0]]
// CHECK:   %[[v3]] -> %[[v1]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @multiple_allocas {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: control_flow_no_phi

// Test: Control flow without phi - different branches use different allocas
// CHECK: amdgcn.kernel @control_flow_no_phi {
// CHECK:   %[[cond:[0-9]*]] = func.call @rand() : () -> i1
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   cf.cond_br %[[cond]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   test_inst ins %[[v0]], %[[v1]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK-DAG:   %[[v0]] -> %[[v0]]
// CHECK-DAG:   %[[v1]] -> %[[v1]]
// CHECK-DAG:   %[[v2]] -> %[[v0]]
// CHECK-DAG:   %[[v3]] -> %[[v1]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1

  kernel @control_flow_no_phi {
    %cond = func.call @rand() : () -> i1
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:
    test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: phi_coalescing_1

// Test: Phi-coalescing - two allocas flow to the same block argument
// CHECK: amdgcn.kernel @phi_coalescing_1 {
// CHECK:   %[[cond:[0-9]*]] = func.call @rand() : () -> i1
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   cf.cond_br %[[cond]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   cf.br ^bb3(%[[v2]] :
// CHECK: ^bb2:
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   cf.br ^bb3(%[[v3]] :
// CHECK: ^bb3(%[[arg:[0-9]*]]:
// CHECK:   test_inst ins %[[arg]]
// CHECK:   end_kernel
// CHECK: }

// Both allocas should be phi-equivalent since they flow to the same block arg
// CHECK: Phi-Equivalences:
// CHECK:   Phi-Equivalent: [%{{[0-9]+}}, %{{[0-9]+}}]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1

  kernel @phi_coalescing_1 {
    %cond = func.call @rand() : () -> i1
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb2:
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3(%3 : !amdgcn.vgpr)
  ^bb3(%arg : !amdgcn.vgpr):
    test_inst ins %arg : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: phi_coalescing_2
// CHECK: amdgcn.kernel @phi_coalescing_2 {
// CHECK:   %[[v0:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]+]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v3:[0-9]+]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v6:[0-9]+]] = test_inst outs %[[v0]] ins %[[v2]]
// CHECK:   %[[v7:[0-9]+]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   %[[vc0:[a-z0-9_]+]] = arith.constant 0 : i32
// CHECK:   %[[v8:[0-9]+]] = lsir.cmpi i32 eq %[[v2]], %[[vc0]]
// CHECK:   cf.cond_br %[[v8]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   cf.br ^bb3(%[[v6]] :
// CHECK: ^bb2:
// CHECK:   cf.br ^bb3(%[[v7]] :
// CHECK: ^bb3(%[[v9:[0-9]+]]:
// CHECK:   test_inst ins %[[v9]], %[[v6]], %[[v7]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v1]]
// CHECK:   %[[v1]] -> %[[v1]]
// CHECK:   %[[v2]] -> %[[v2]]
// CHECK:   %[[v3]] -> %[[v3]]
// CHECK:   %[[v4]] -> %[[v4]]
// CHECK:   %[[v5]] -> %[[v5]]
// CHECK:   %[[v6]] -> %[[v1]]
// CHECK:   %[[v7]] -> %[[v1]]
// CHECK:   %[[vc0]] -> <unknown>
// CHECK:   %[[v8]] -> <unknown>

// CHECK: Phi-Equivalences:
// CHECK:   Phi-Equivalent: [%[[v1]], %[[v0]]]

amdgcn.module @ra target = <gfx942> isa = <cdna3> {
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
    cf.br ^bb3(%7 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%8 : !amdgcn.vgpr)
  ^bb3(%val: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    test_inst ins %val, %7, %8 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: cf_bbargs_and_lsir_cmpi
// CHECK: amdgcn.kernel @cf_bbargs_and_lsir_cmpi {
// CHECK:   %[[v0:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v2:[0-9]+]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v3:[0-9]+]] = alloca : !amdgcn.sgpr
// CHECK:   %[[v4:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v5:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v6:[0-9]+]] = test_inst outs %[[v0]] ins %[[v2]]
// CHECK:   %[[v7:[0-9]+]] = test_inst outs %[[v1]] ins %[[v3]]
// CHECK:   %[[vc0:[a-z0-9_]+]] = arith.constant 0 : i32
// CHECK:   %[[v8:[0-9]+]] = lsir.cmpi i32 eq %[[v2]], %[[vc0]]
// CHECK:   cf.cond_br %[[v8]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[v9:[0-9]+]] = test_inst outs %[[v4]] ins %[[v6]]
// CHECK:   %[[v10:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v11:[0-9]+]] = test_inst outs %[[v10]]
// CHECK:   cf.br ^bb3(%[[v11]] :
// CHECK: ^bb2:
// CHECK:   %[[v12:[0-9]+]] = test_inst outs %[[v5]] ins %[[v7]]
// CHECK:   %[[v13:[0-9]+]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v14:[0-9]+]] = test_inst outs %[[v13]]
// CHECK:   cf.br ^bb3(%[[v14]] :
// CHECK: ^bb3(%[[v15:[0-9]+]]:
// CHECK:   test_inst ins %[[v15]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]
// CHECK:   %[[v1]] -> %[[v1]]
// CHECK:   %[[v2]] -> %[[v2]]
// CHECK:   %[[v3]] -> %[[v3]]
// CHECK:   %[[v4]] -> %[[v4]]
// CHECK:   %[[v5]] -> %[[v5]]
// CHECK:   %[[v6]] -> %[[v0]]
// CHECK:   %[[v7]] -> %[[v1]]
// CHECK:   %[[vc0]] -> <unknown>
// CHECK:   %[[v8]] -> <unknown>
// CHECK:   %[[v9]] -> %[[v4]]
// CHECK:   %[[v10]] -> %[[v13]]
// CHECK:   %[[v11]] -> %[[v13]]
// CHECK:   %[[v12]] -> %[[v5]]
// CHECK:   %[[v13]] -> %[[v13]]
// CHECK:   %[[v14]] -> %[[v13]]

// CHECK: Phi-Equivalences:
// CHECK:   Phi-Equivalent: [%[[v13]], %[[v10]]]

amdgcn.module @ra target = <gfx942> isa = <cdna3> {
  kernel @cf_bbargs_and_lsir_cmpi {
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

// Test: phi coalescing - allocas in different branches flowing to same block arg
// are detected as phi-equivalent.
// CHECK: Kernel: phi_coalescing_2
// CHECK: Value Provenances:
// CHECK-DAG:   %0 -> %0
// CHECK-DAG:   %1 -> %1
// CHECK-DAG:   %10 -> %13
// CHECK-DAG:   %13 -> %13
// CHECK: Phi-Equivalences:
// Allocas %10 and %13 are phi-equivalent (they flow to same block arg)
// CHECK:   Phi-Equivalent: [%13, %10]
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

// Test: phi coalescing - values %6 and %7 flow to same block arg, so their source
// allocas %0 and %1 are phi-equivalent.
// CHECK: Kernel: phi_coalescing_3
// CHECK: Value Provenances:
// Both %0 and %1 trace to the same canonical alloca (%1) due to phi-equivalence
// CHECK-DAG:   %0 -> %1
// CHECK-DAG:   %1 -> %1
// CHECK-DAG:   %6 -> %1
// CHECK-DAG:   %7 -> %1
// CHECK: Phi-Equivalences:
// CHECK:   Phi-Equivalent: [%1, %0]
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

// Test: same as phi_coalescing_3 - phi coalescing detection.
// CHECK: Kernel: phi_coalescing_4
// CHECK: Value Provenances:
// CHECK-DAG:   %0 -> %1
// CHECK-DAG:   %1 -> %1
// CHECK-DAG:   %6 -> %1
// CHECK-DAG:   %7 -> %1
// CHECK: Phi-Equivalences:
// CHECK:   Phi-Equivalent: [%1, %0]
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

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: non_dps_value

// Test: Non-alloca values get unknown provenance
// CHECK: amdgcn.kernel @non_dps_value {
// CHECK:   %[[v0:[0-9]*]] = func.call @get_value() : () -> i32
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> <unknown>

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  func.func private @get_value() -> i32

  kernel @non_dps_value {
    %0 = func.call @get_value() : () -> i32
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: empty_kernel

// Test: Empty kernel with no allocas
// CHECK: amdgcn.kernel @empty_kernel {
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK-NOT: %

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @empty_kernel {
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: single_alloca_unused

// Test: Single alloca with no uses
// CHECK: amdgcn.kernel @single_alloca_unused {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @single_alloca_unused {
    %0 = alloca : !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: deep_dps_chain

// Test: Deep chain of 4 operations - provenance flows through all
// CHECK: amdgcn.kernel @deep_dps_chain {
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   %[[v2:[0-9]*]] = test_inst outs %[[v1]]
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v2]]
// CHECK:   %[[v4:[0-9]*]] = test_inst outs %[[v3]]
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK:   %[[v0]] -> %[[v0]]
// CHECK:   %[[v1]] -> %[[v0]]
// CHECK:   %[[v2]] -> %[[v0]]
// CHECK:   %[[v3]] -> %[[v0]]
// CHECK:   %[[v4]] -> %[[v0]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  kernel @deep_dps_chain {
    %0 = alloca : !amdgcn.vgpr
    %1 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %2 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK: === Value Provenance Analysis Results ===
// CHECK-LABEL: Kernel: allocas_in_branches

// Test: Allocas defined inside branches (not in entry block)
// CHECK: amdgcn.kernel @allocas_in_branches {
// CHECK:   %[[cond:[0-9]*]] = func.call @rand() : () -> i1
// CHECK:   cf.cond_br %[[cond]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[v0:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v1:[0-9]*]] = test_inst outs %[[v0]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb2:
// CHECK:   %[[v2:[0-9]*]] = alloca : !amdgcn.vgpr
// CHECK:   %[[v3:[0-9]*]] = test_inst outs %[[v2]]
// CHECK:   cf.br ^bb3
// CHECK: ^bb3:
// CHECK:   end_kernel
// CHECK: }

// CHECK: Value Provenances:
// CHECK-DAG:   %[[v0]] -> %[[v0]]
// CHECK-DAG:   %[[v1]] -> %[[v0]]
// CHECK-DAG:   %[[v2]] -> %[[v2]]
// CHECK-DAG:   %[[v3]] -> %[[v2]]

amdgcn.module @provenance_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1

  kernel @allocas_in_branches {
    %cond = func.call @rand() : () -> i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %0 = alloca : !amdgcn.vgpr
    %1 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb2:
    %2 = alloca : !amdgcn.vgpr
    %3 = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.br ^bb3
  ^bb3:
    end_kernel
  }
}
