// RUN: aster-opt %s --test-amdgcn-interference-analysis=optimize=true --split-input-file 2>&1 | FileCheck %s

// ============================================================================
// Test 1: Basic two-stage pipelined load (same pattern as scf-pipeline-two-stage)
// Prologue: load(iter 0), copy dest -> shadow
// Kernel: use(shadow), load(iter i), copy dest -> shadow, branch
// Epilogue: use(shadow)
// The copy shadow <- dest should coalesce (nodes 2,3) since they never
// interfere: dest is dead by the time shadow is consumed.
// ============================================================================
// CHECK-LABEL: Function: two_stage_load_basic
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   0 -- 2;
// CHECK:   1 -- 2;
// CHECK: }
amdgcn.module @t1 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @two_stage_load_basic {
    %cond = func.call @rand() : () -> i1
    %addr0 = alloca : !amdgcn.vgpr<?>
    %addr1 = alloca : !amdgcn.vgpr<?>
    %dest = alloca : !amdgcn.vgpr<?>
    %dest_copy = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %addr0, %addr1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %dest_copy, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %dest_copy : (!amdgcn.vgpr<?>) -> ()
    %tok1 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %dest_copy, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst ins %dest_copy : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 2: Two independent loads in a two-stage pipeline
// Both loads have separate shadow copies. Both copies should coalesce
// independently: [2,3] for load A and [4,5] for load B.
// Pattern from scf-pipeline-multi-value: multiple cross-stage values.
// ============================================================================
// CHECK-LABEL: Function: two_independent_loads
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   2 -- 4;
// CHECK:   3 -- 4;
// CHECK:   3 -- 5;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3]
// CHECK:   [4, 5]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   4 [label="4, %5"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 4;
// CHECK:   1 -- 2;
// CHECK:   1 -- 4;
// CHECK:   2 -- 4;
// CHECK: }
amdgcn.module @t2 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @two_independent_loads {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %dest1 = alloca : !amdgcn.vgpr<?>
    %copy1 = alloca : !amdgcn.vgpr<?>
    %dest2 = alloca : !amdgcn.vgpr<?>
    %copy2 = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tokA = load global_load_dword dest %dest1 addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokB = load global_load_dword dest %dest2 addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy1, %dest1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copy2, %dest2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %copy1, %copy2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %tokC = load global_load_dword dest %dest1 addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokD = load global_load_dword dest %dest2 addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy1, %dest1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copy2, %dest2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst ins %copy1, %copy2 : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 3: dwordx2 load with range copy (2-wide register range)
// Both elements of the dest range coalesce pairwise with the copy range:
// [2,4] and [3,5]. Tests range-aware coalescing from load -> range copy.
// ============================================================================
// CHECK-LABEL: Function: two_stage_load_dwordx2
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   2 -- 3;
// CHECK:   4 -- 5;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 4]
// CHECK:   [3, 5]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   2 -- 3;
// CHECK: }
amdgcn.module @t3 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @two_stage_load_dwordx2 {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %d0 = alloca : !amdgcn.vgpr<?>
    %d1 = alloca : !amdgcn.vgpr<?>
    %c0 = alloca : !amdgcn.vgpr<?>
    %c1 = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %dest = make_register_range %d0, %d1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %copy = make_register_range %c0, %c1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dwordx2 dest %dest addr %addr : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
    cf.br ^loop
  ^loop:
    test_inst ins %copy : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    %tok1 = load global_load_dwordx2 dest %dest addr %addr : dps(!amdgcn.vgpr<[? : ? + 2]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<[? : ? + 2]>, !amdgcn.vgpr<[? : ? + 2]>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst ins %copy : (!amdgcn.vgpr<[? : ? + 2]>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 4: Three-stage pipeline: load -> compute -> store
// dest and shadow interfere because shadow is consumed by compute while
// dest is being reloaded. No copies can coalesce.
// Pattern from scf-pipeline-three-stage with explicit prologue/kernel/epilogue.
// ============================================================================
// CHECK-LABEL: Function: three_stage_load_compute_store
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
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
// CHECK-NOT: EquivalenceClasses
amdgcn.module @t4 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @three_stage_load_compute_store {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %dest = alloca : !amdgcn.vgpr<?>
    %shadow = alloca : !amdgcn.vgpr<?>
    %comp = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %shadow, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok1 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    test_inst outs %comp ins %shadow : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    lsir.copy %shadow, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %comp : (!amdgcn.vgpr<?>) -> ()
    %tok2 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    test_inst outs %comp ins %shadow : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    lsir.copy %shadow, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst ins %comp : (!amdgcn.vgpr<?>) -> ()
    test_inst outs %comp ins %shadow : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    test_inst ins %comp : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 5: Pipeline with copy that CANNOT coalesce due to interference
// test_inst uses both %copy and %dest simultaneously, creating interference
// edge 2--3. The copy cannot be eliminated.
// ============================================================================
// CHECK-LABEL: Function: pipeline_copy_interferes
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   2 -- 3;
// CHECK: }
// CHECK-NOT: EquivalenceClasses
amdgcn.module @t5 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @pipeline_copy_interferes {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %dest = alloca : !amdgcn.vgpr<?>
    %copy = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %copy, %dest : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %tok1 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^exit
  ^exit:
    end_kernel
  }
}

// -----

// ============================================================================
// Test 6: Mixed coalescing -- one copy coalesces, the other does not
// Load A: copyA used alone -> destA/copyA coalesce [2,3]
// Load B: copyB and destB used together -> they interfere (edge 4--5),
// so no coalescing for load B.
// ============================================================================
// CHECK-LABEL: Function: mixed_coalesce_and_no_coalesce
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   2 -- 4;
// CHECK:   3 -- 4;
// CHECK:   3 -- 5;
// CHECK:   4 -- 5;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3]
// CHECK:   [4]
// CHECK:   [5]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   1 -- 2;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   2 -- 4;
// CHECK:   2 -- 5;
// CHECK:   4 -- 5;
// CHECK: }
amdgcn.module @t6 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @mixed_coalesce_and_no_coalesce {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %destA = alloca : !amdgcn.vgpr<?>
    %copyA = alloca : !amdgcn.vgpr<?>
    %destB = alloca : !amdgcn.vgpr<?>
    %copyB = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tokA = load global_load_dword dest %destA addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokB = load global_load_dword dest %destB addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copyA, %destA : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copyB, %destB : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %copyA : (!amdgcn.vgpr<?>) -> ()
    test_inst ins %copyB, %destB : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %tokC = load global_load_dword dest %destA addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokD = load global_load_dword dest %destB addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copyA, %destA : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copyB, %destB : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^exit
  ^exit:
    end_kernel
  }
}

// -----

// ============================================================================
// Test 7: Chain of copies (load -> dest -> mid -> final)
// Simulates a deeper pipeline with shift-register copies.
// All three (dest, mid, final) coalesce into [2, 3, 4] since none interfere.
// ============================================================================
// CHECK-LABEL: Function: pipeline_chain_copies
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3, 4]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   0 -- 2;
// CHECK:   1 -- 2;
// CHECK: }
amdgcn.module @t7 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @pipeline_chain_copies {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %dest = alloca : !amdgcn.vgpr<?>
    %mid = alloca : !amdgcn.vgpr<?>
    %final = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %mid, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %final, %mid : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst ins %final : (!amdgcn.vgpr<?>) -> ()
    %tok1 = load global_load_dword dest %dest addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %mid, %dest : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %final, %mid : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^exit
  ^exit:
    test_inst ins %final : (!amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 8: dwordx4 load with 4-wide range copy
// All 4 elements coalesce pairwise: [2,6], [3,7], [4,8], [5,9].
// Tests range-aware coalescing with wider register ranges.
// ============================================================================
// CHECK-LABEL: Function: two_stage_load_dwordx4
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   6 [label="6, %7"];
// CHECK:   7 [label="7, %8"];
// CHECK:   8 [label="8, %9"];
// CHECK:   9 [label="9, %10"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   0 -- 6;
// CHECK:   0 -- 7;
// CHECK:   0 -- 8;
// CHECK:   0 -- 9;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   1 -- 6;
// CHECK:   1 -- 7;
// CHECK:   1 -- 8;
// CHECK:   1 -- 9;
// CHECK:   2 -- 3;
// CHECK:   2 -- 4;
// CHECK:   2 -- 5;
// CHECK:   3 -- 4;
// CHECK:   3 -- 5;
// CHECK:   4 -- 5;
// CHECK:   6 -- 7;
// CHECK:   6 -- 8;
// CHECK:   6 -- 9;
// CHECK:   7 -- 8;
// CHECK:   7 -- 9;
// CHECK:   8 -- 9;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 6]
// CHECK:   [3, 7]
// CHECK:   [4, 8]
// CHECK:   [5, 9]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   2 -- 3;
// CHECK:   2 -- 4;
// CHECK:   2 -- 5;
// CHECK:   3 -- 4;
// CHECK:   3 -- 5;
// CHECK:   4 -- 5;
// CHECK: }
amdgcn.module @t8 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @two_stage_load_dwordx4 {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %d0 = alloca : !amdgcn.vgpr<?>
    %d1 = alloca : !amdgcn.vgpr<?>
    %d2 = alloca : !amdgcn.vgpr<?>
    %d3 = alloca : !amdgcn.vgpr<?>
    %c0 = alloca : !amdgcn.vgpr<?>
    %c1 = alloca : !amdgcn.vgpr<?>
    %c2 = alloca : !amdgcn.vgpr<?>
    %c3 = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %dest = make_register_range %d0, %d1, %d2, %d3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %copy = make_register_range %c0, %c1, %c2, %c3 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tok0 = load global_load_dwordx4 dest %dest addr %addr : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[? : ? + 4]>
    cf.br ^loop
  ^loop:
    test_inst ins %copy : (!amdgcn.vgpr<[? : ? + 4]>) -> ()
    %tok1 = load global_load_dwordx4 dest %dest addr %addr : dps(!amdgcn.vgpr<[? : ? + 4]>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copy, %dest : !amdgcn.vgpr<[? : ? + 4]>, !amdgcn.vgpr<[? : ? + 4]>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst ins %copy : (!amdgcn.vgpr<[? : ? + 4]>) -> ()
    end_kernel
  }
}

// -----

// ============================================================================
// Test 9: Two loads feeding a compute op, both with shadow copies
// Both load-sourced copies (priority 0) coalesce: [2,3] and [4,5].
// The compute output %comp (node 6) is separate and does not participate
// in coalescing.
// ============================================================================
// CHECK-LABEL: Function: two_loads_to_compute
// CHECK: graph RegisterInterference {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   3 [label="3, %4"];
// CHECK:   4 [label="4, %5"];
// CHECK:   5 [label="5, %6"];
// CHECK:   6 [label="6, %7"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 3;
// CHECK:   0 -- 4;
// CHECK:   0 -- 5;
// CHECK:   0 -- 6;
// CHECK:   1 -- 2;
// CHECK:   1 -- 3;
// CHECK:   1 -- 4;
// CHECK:   1 -- 5;
// CHECK:   1 -- 6;
// CHECK:   2 -- 4;
// CHECK:   3 -- 4;
// CHECK:   3 -- 5;
// CHECK: }
// CHECK: EquivalenceClasses {
// CHECK:   [0]
// CHECK:   [1]
// CHECK:   [2, 3]
// CHECK:   [4, 5]
// CHECK:   [6]
// CHECK: }
// CHECK: graph RegisterInterferenceQuotient {
// CHECK:   0 [label="0, %1"];
// CHECK:   1 [label="1, %2"];
// CHECK:   2 [label="2, %3"];
// CHECK:   4 [label="4, %5"];
// CHECK:   6 [label="6, %7"];
// CHECK:   0 -- 2;
// CHECK:   0 -- 4;
// CHECK:   0 -- 6;
// CHECK:   1 -- 2;
// CHECK:   1 -- 4;
// CHECK:   1 -- 6;
// CHECK:   2 -- 4;
// CHECK: }
amdgcn.module @t9 target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @two_loads_to_compute {
    %cond = func.call @rand() : () -> i1
    %a0 = alloca : !amdgcn.vgpr<?>
    %a1 = alloca : !amdgcn.vgpr<?>
    %destA = alloca : !amdgcn.vgpr<?>
    %copyA = alloca : !amdgcn.vgpr<?>
    %destB = alloca : !amdgcn.vgpr<?>
    %copyB = alloca : !amdgcn.vgpr<?>
    %comp = alloca : !amdgcn.vgpr<?>
    %addr = make_register_range %a0, %a1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    %tokA0 = load global_load_dword dest %destA addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokB0 = load global_load_dword dest %destB addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copyA, %destA : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copyB, %destB : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.br ^loop
  ^loop:
    test_inst outs %comp ins %copyA, %copyB : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    %tokA1 = load global_load_dword dest %destA addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    %tokB1 = load global_load_dword dest %destB addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
    lsir.copy %copyA, %destA : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    lsir.copy %copyB, %destB : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
    cf.cond_br %cond, ^loop, ^epilogue
  ^epilogue:
    test_inst outs %comp ins %copyA, %copyB : (!amdgcn.vgpr<?>, !amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> ()
    end_kernel
  }
}
