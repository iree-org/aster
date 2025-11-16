// RUN: aster-opt %s --verify-diagnostics --split-input-file

func.func @test_add(%dst: !pir.reg<f32 : !amdgcn.sgpr>, %lhs: !pir.reg<i32 : !amdgcn.sgpr>, %rhs: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<f32 : !amdgcn.sgpr>{
  // expected-error@+1 {{op operand #0 must be Integer Register, but got '!pir.reg<f32 : !amdgcn.sgpr>'}}
  %0 = pir.addi %dst, %lhs, %rhs : !pir.reg<f32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<f32 : !amdgcn.sgpr>
}
