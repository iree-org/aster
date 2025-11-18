// RUN: aster-opt %s --verify-diagnostics --split-input-file

func.func @test_add(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  // expected-error@+1 {{op attribute 'semantics' failed to satisfy constraint: any integer type attribute}}
  %0 = lsir.addi f32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

// -----

// Test: duplicate operands
func.func @test_duplicate_operands(
    %ptr1: !amdgcn.sgpr_range<[? + 2]>,
    %ptr2: !amdgcn.sgpr_range<[? + 2]>) {
  // expected-error@+1 {{'lsir.assume_noalias' op operand 0 and operand 1 must be different}}
  %ptr1_noalias, %ptr2_noalias = lsir.assume_noalias %ptr1, %ptr1
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
    -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
  return
}

// -----

// Test: mismatched operand/result counts
func.func @test_mismatched_counts(
    %ptr1: !amdgcn.sgpr_range<[? + 2]>,
    %ptr2: !amdgcn.sgpr_range<[? + 2]>,
    %ptr3: !amdgcn.sgpr_range<[? + 2]>) {
  // expected-error@+1 {{'lsir.assume_noalias' op number of operands (3) must match number of results (2)}}
  %ptr1_noalias, %ptr2_noalias = lsir.assume_noalias %ptr1, %ptr2, %ptr3
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
    -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
  return
}

// -----

// Test: mismatched types
func.func @test_mismatched_types(
    %sgpr: !amdgcn.sgpr_range<[? + 2]>,
    %vgpr: !amdgcn.vgpr_range<[? + 4]>) {
  // expected-error@+1 {{'lsir.assume_noalias' op operand 0 type '!amdgcn.sgpr_range<[? + 2]>' must match result 0 type '!amdgcn.vgpr_range<[? + 4]>'}}
  %sgpr_noalias, %vgpr_noalias = lsir.assume_noalias %sgpr, %vgpr
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>)
    -> (!amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? + 4]>)
  return
}
