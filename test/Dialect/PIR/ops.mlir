// RUN: aster-opt %s --verify-roundtrip

func.func @test_add(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.addi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_sub(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.subi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_mul(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.muli i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_divsi(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.divsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_divui(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.divui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_remsi(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.remsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_remui(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.remui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_and(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.andi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_or(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.ori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_shli(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.shli i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_shrsi(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.shrsi i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_shrui(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.shrui i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_eq(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.cmpi i32 eq %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_cmpi_ne(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.cmpi i32 ne %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_cmpi_slt(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 slt %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_sle(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 sle %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_sgt(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 sgt %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_sge(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 sge %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_ult(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 ult %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_ule(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 ule %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_ugt(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 ugt %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_uge(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.cmpi i32 uge %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_extsi(%dst: !amdgcn.sgpr, %value: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.extsi i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_extui(%dst: !amdgcn.sgpr, %value: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = pir.extui i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_trunci(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.trunci i32 from i16 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_mov(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = pir.mov %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}
