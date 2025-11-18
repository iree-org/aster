// RUN: aster-opt %s --verify-diagnostics --split-input-file

func.func @test_add(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  // expected-error@+1 {{op attribute 'semantics' failed to satisfy constraint: type attribute of signless integer}}
  %0 = pir.addi f32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}
