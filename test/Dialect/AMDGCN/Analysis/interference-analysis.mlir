// RUN: aster-opt %s --test-interference-analysis --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: // Kernel: no_interference
// CHECK: graph InterferenceAnalysis {
// No edges expected - values don't interfere
// CHECK: }

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @no_interference {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @basic_interference {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @three_way_interference {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @no_cross_type_interference {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @reg_interference_op {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @partial_interference {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @diamond_cf {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @live_across_diamond {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @sequential_use {
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

amdgcn.module @test target = <gfx942> isa = <cdna3> {
  kernel @many_overlapping {
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
