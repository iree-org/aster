// RUN: aster-opt %s --verify-roundtrip

func.func @test_add(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.addi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_sub(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.subi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_mul(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.muli %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_divsi(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.divsi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_divui(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.divui %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_remsi(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.remsi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_remui(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.remui %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_and(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %lhs: !pir.reg<i32 : !amdgcn.sgpr>, %rhs: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.andi %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_or(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %lhs: !pir.reg<i32 : !amdgcn.sgpr>, %rhs: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.ori %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_shli(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<i32 : !amdgcn.vgpr>, %amount: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.shli %dst, %value, %amount : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_shrsi(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<i32 : !amdgcn.vgpr>, %amount: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.shrsi %dst, %value, %amount : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_shrui(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<i32 : !amdgcn.vgpr>, %amount: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.shrui %dst, %value, %amount : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_eq(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %lhs: !pir.reg<i32 : !amdgcn.sgpr>, %rhs: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.cmpi eq %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_cmpi_ne(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %lhs: !pir.reg<i32 : !amdgcn.sgpr>, %rhs: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.cmpi ne %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_cmpi_slt(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi slt %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_sle(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi sle %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_sgt(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi sgt %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_sge(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi sge %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_ult(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi ult %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_ule(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi ule %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_ugt(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi ugt %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_cmpi_uge(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %lhs: !pir.reg<i32 : !amdgcn.vgpr>, %rhs: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.cmpi uge %dst, %lhs, %rhs : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_extsi(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %value: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.extsi %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_extui(%dst: !pir.reg<i32 : !amdgcn.sgpr>, %value: !pir.reg<i32 : !amdgcn.sgpr>) -> !pir.reg<i32 : !amdgcn.sgpr> {
  %0 = pir.extui %dst, %value : !pir.reg<i32 : !amdgcn.sgpr>, !pir.reg<i32 : !amdgcn.sgpr>
  return %0 : !pir.reg<i32 : !amdgcn.sgpr>
}

func.func @test_trunci(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.trunci %dst, %value : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_mov(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<i32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.mov %dst, %value : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<i32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}

func.func @test_type_cast(%dst: !pir.reg<i32 : !amdgcn.vgpr>, %value: !pir.reg<f32 : !amdgcn.vgpr>) -> !pir.reg<i32 : !amdgcn.vgpr> {
  %0 = pir.type_cast %dst, %value : !pir.reg<i32 : !amdgcn.vgpr>, !pir.reg<f32 : !amdgcn.vgpr>
  return %0 : !pir.reg<i32 : !amdgcn.vgpr>
}
