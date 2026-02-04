// RUN: aster-opt %s --test-interference-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: // Kernel: no_interference
// CHECK: graph InterferenceAnalysis {
// No edges expected - values don't interfere
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @no_interference {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: basic_interference
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// Both values are live at the final use, so they interfere
// CHECK: 0 -- 1;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @basic_interference {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // Both %2 and %3 are live here - they interfere
    test_inst ins %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: three_way_interference
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// CHECK-DAG: 2 [label="2,
// All three values are live at the final use
// CHECK-DAG: 0 -- 1;
// CHECK-DAG: 0 -- 2;
// CHECK-DAG: 1 -- 2;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @three_way_interference {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %a = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %c = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // All three are live - full clique
    test_inst ins %a, %b, %c : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: no_cross_type_interference
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// VGPRs and SGPRs don't interfere with each other (different resource types)
// CHECK-NOT: 0 -- 1;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @no_cross_type_interference {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.sgpr
    %a = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = test_inst outs %1 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    // VGPR and SGPR live together but don't interfere
    test_inst ins %a, %b : (!amdgcn.vgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: reg_interference_op
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// CHECK-DAG: 2 [label="2,
// reg_interference forces interference between its operands
// CHECK-DAG: 0 -- 1;
// CHECK-DAG: 0 -- 2;
// CHECK-DAG: 1 -- 2;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @reg_interference_op {
    %0 = alloca : !amdgcn.sgpr
    %1 = alloca : !amdgcn.sgpr
    %2 = alloca : !amdgcn.sgpr
    %a = test_inst outs %0 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %b = test_inst outs %1 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    %c = test_inst outs %2 : (!amdgcn.sgpr) -> !amdgcn.sgpr
    // Force interference between these values
    amdgcn.reg_interference %a, %b, %c : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: partial_interference
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// CHECK-DAG: 2 [label="2,
// All allocas interfere at the allocation level (conservative analysis)
// CHECK-DAG: 0 -- 1;
// CHECK-DAG: 0 -- 2;
// CHECK-DAG: 1 -- 2;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @partial_interference {
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    %a = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // %a and %b both live here - they interfere
    test_inst ins %a, %b : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    // Interference is computed at alloca level (conservative), so all allocas interfere
    %c = test_inst outs %2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %c : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: diamond_cf
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// Both allocas exist before the branch - conservative analysis marks them as interfering
// CHECK: 0 -- 1;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  amdgcn.kernel @diamond_cf {
    %cond = func.call @rand() : () -> i1
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %a = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %a : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb2:
    %b = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %b : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb3:
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: live_across_diamond
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// CHECK-DAG: 2 [label="2,
// %pre is live across the diamond, interferes with both branch values
// CHECK-DAG: 0 -- 1;
// CHECK-DAG: 0 -- 2;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  amdgcn.kernel @live_across_diamond {
    %cond = func.call @rand() : () -> i1
    %s0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.vgpr
    %s2 = alloca : !amdgcn.vgpr
    %pre = test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %a = test_inst outs %s1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %a : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb2:
    %b = test_inst outs %s2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %b : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb3:
    // %pre is used here, so it was live through both branches
    test_inst ins %pre : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: sequential_use
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG: 0 [label="0,
// CHECK-DAG: 1 [label="1,
// Both allocas exist at the same time - conservative analysis marks interference
// CHECK: 0 -- 1;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @sequential_use {
    %s0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.vgpr
    %a = test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %a : (!amdgcn.vgpr) -> ()
    // %a is now dead
    %b = test_inst outs %s1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %b : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// CHECK-LABEL: // Kernel: many_overlapping
// CHECK: graph InterferenceAnalysis {
// All 5 values interfere - should be a complete graph (K5)
// CHECK-DAG: 0 -- 1;
// CHECK-DAG: 0 -- 2;
// CHECK-DAG: 0 -- 3;
// CHECK-DAG: 0 -- 4;
// CHECK-DAG: 1 -- 2;
// CHECK-DAG: 1 -- 3;
// CHECK-DAG: 1 -- 4;
// CHECK-DAG: 2 -- 3;
// CHECK-DAG: 2 -- 4;
// CHECK-DAG: 3 -- 4;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @many_overlapping {
    %s0 = alloca : !amdgcn.vgpr
    %s1 = alloca : !amdgcn.vgpr
    %s2 = alloca : !amdgcn.vgpr
    %s3 = alloca : !amdgcn.vgpr
    %s4 = alloca : !amdgcn.vgpr
    %a = test_inst outs %s0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = test_inst outs %s1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %c = test_inst outs %s2 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %d = test_inst outs %s3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %e = test_inst outs %s4 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %a, %b, %c, %d, %e : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}
// -----

// Test: phi coalescing - allocas in different branches that flow to the same block
// argument get phi-coalesced into the same equivalence class (eq class 6).
//
// 7 equivalence classes total:
// - EqClass 0: first VGPR alloca and its result
// - EqClass 1: second VGPR alloca and its result
// - EqClass 2: first SGPR alloca
// - EqClass 3: second SGPR alloca
// - EqClass 4: third VGPR alloca (bb1 use) and its result
// - EqClass 5: fourth VGPR alloca (bb2 use) and its result
// - EqClass 6: phi-coalesced allocas from bb1 and bb2 (alloc0 and alloc1)
//
// CHECK-LABEL: // Kernel: phi_coalescing_2
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG:   0 [label="0, %[[eq0:[0-9]+]]"];
// CHECK-DAG:   1 [label="1, %[[eq1:[0-9]+]]"];
// CHECK-DAG:   2 [label="2, %[[eq2:[0-9]+]]"];
// CHECK-DAG:   3 [label="3, %[[eq3:[0-9]+]]"];
// CHECK-DAG:   4 [label="4, %[[eq4:[0-9]+]]"];
// CHECK-DAG:   5 [label="5, %[[eq5:[0-9]+]]"];
// Phi-coalesced allocas from different branches share eq class 6
// CHECK-DAG:   6 [label="6, %[[eq6:[0-9]+]]"];
// VGPRs 0 and 1 interfere (both computed in entry block, used later)
// CHECK-DAG:   0 -- 1;
// SGPRs 2 and 3 interfere (both live at the same time)
// CHECK-DAG:   2 -- 3;
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @phi_coalescing_2 {
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
// allocas %0 and %1 get phi-coalesced into the same equivalence class.
//
// Only 5 equivalence classes (instead of 6) due to phi-coalescing:
// - EqClass 0: phi-coalesced VGPRs (allocas whose results flow to same block arg)
// - EqClass 1: first SGPR alloca
// - EqClass 2: second SGPR alloca
// - EqClass 3: unused VGPR alloca
// - EqClass 4: unused VGPR alloca
//
// CHECK-LABEL: // Kernel: phi_coalescing_3
// CHECK: graph InterferenceAnalysis {
// CHECK-DAG:   0 [label="0, %[[eq0:[0-9]+]]"];
// CHECK-DAG:   1 [label="1, %[[eq1:[0-9]+]]"];
// CHECK-DAG:   2 [label="2, %[[eq2:[0-9]+]]"];
// CHECK-DAG:   3 [label="3, %[[eq3:[0-9]+]]"];
// CHECK-DAG:   4 [label="4, %[[eq4:[0-9]+]]"];
// SGPRs 1 and 2 interfere (both live at the same time)
// CHECK-DAG:   1 -- 2;
// No eq class 5 - allocas were phi-coalesced into eq class 0
// CHECK-NOT:   5 [label=
// CHECK: }

amdgcn.module @interference_tests target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @phi_coalescing_3 {
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
