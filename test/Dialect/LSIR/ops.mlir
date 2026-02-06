// RUN: aster-opt %s --verify-roundtrip

func.func @test_add(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.addi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_sub(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.subi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_mul(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.muli i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_divsi(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.divsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_divui(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.divui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_remsi(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.remsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_remui(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.remui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_and(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.andi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_or(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.ori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_shli(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.shli i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_shrsi(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.shrsi i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_shrui(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr, %amount: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.shrui i32 %dst, %value, %amount : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_cmpi_eq(%lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> i1 {
  %0 = lsir.cmpi i32 eq %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : i1
}

func.func @test_cmpi_ne(%lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> i1 {
  %0 = lsir.cmpi i32 ne %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : i1
}

func.func @test_cmpi_slt(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 slt %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_sle(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 sle %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_sgt(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 sgt %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_sge(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 sge %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_ult(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 ult %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_ule(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 ule %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_ugt(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 ugt %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpi_uge(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpi i32 uge %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_extsi(%dst: !amdgcn.sgpr, %value: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.extsi i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_extui(%dst: !amdgcn.sgpr, %value: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.extui i32 from i16 %dst, %value : !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_trunci(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.trunci i32 from i16 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_mov(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.mov %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_timing_start() -> !amdgcn.sgpr_range<[? + 2]> {
  %start_time = lsir.timing_start : !amdgcn.sgpr_range<[? + 2]>
  return %start_time : !amdgcn.sgpr_range<[? + 2]>
}

func.func @test_timing_stop(%start_time: !amdgcn.sgpr_range<[? + 2]>, %start_buffer: !amdgcn.sgpr_range<[? + 2]>, %end_buffer: !amdgcn.sgpr_range<[? + 2]>) {
  lsir.timing_stop %start_time, %start_buffer, %end_buffer
    : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>
  return
}

func.func @test_assume_noalias_mixed_types(
    %sgpr: !amdgcn.sgpr_range<[? + 2]>,
    %sgpr_2: !amdgcn.sgpr_range<[? + 2]>,
    %vgpr: !amdgcn.vgpr_range<[? + 4]>) {
  %sgpr_noalias, %sgpr_noalias_2, %vgpr_noalias = lsir.assume_noalias %sgpr, %sgpr_2, %vgpr
    : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>)
    -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>)
  return
}

func.func @test_addf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.addf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_subf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.subf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_mulf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.mulf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_divf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.divf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_negf(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.negf f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_maximumf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.maximumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_minimumf(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.minimumf f32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_maxsi(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.maxsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_maxui(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.maxui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_xori(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.xori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_cmpf_olt(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpf f32 olt %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_cmpf_oeq(%lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> i1 {
  %0 = lsir.cmpf f32 oeq %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : i1
}

func.func @test_extf(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.extf f64 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_truncf(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.truncf f32 from f64 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_fptosi(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.fptosi i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_fptoui(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.fptoui i32 from f32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_sitofp(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.sitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_uitofp(%dst: !amdgcn.vgpr, %value: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.uitofp f32 from i32 %dst, %value : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_select(%dst: !amdgcn.vgpr, %cond: !amdgcn.vgpr, %true_val: !amdgcn.vgpr, %false_val: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.select %dst, %cond, %true_val, %false_val : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// ...existing code...

func.func @test_load_global(%dst: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 1]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_write> %dst, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr_range<[? + 1]>
}

func.func @test_load_global_with_offset(%dst: !amdgcn.vgpr_range<[? + 4]>, %addr: !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 4]> {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_write> %dst, %addr, %c0, %c16 : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr_range<[? + 4]>
}

func.func @test_load_local(%dst: !amdgcn.vgpr_range<[? + 2]>, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr_range<[? + 2]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_write> %dst, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr_range<[? + 2]>
}

func.func @test_load_smem(%dst: !amdgcn.sgpr_range<[? + 4]>, %addr: !amdgcn.sgpr_range<[? + 2]>) -> !amdgcn.sgpr_range<[? + 4]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr_range<[? + 4]>
}

func.func @test_store_global(%data: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  return
}

func.func @test_store_global_with_offset(%data: !amdgcn.vgpr_range<[? + 4]>, %addr: !amdgcn.vgpr_range<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c32 : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  return
}

func.func @test_store_local(%data: !amdgcn.vgpr_range<[? + 2]>, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32, i32
  return
}

func.func @test_store_smem(%data: !amdgcn.sgpr_range<[? + 4]>, %addr: !amdgcn.sgpr_range<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, i32, i32
  return
}

func.func @test_wait_single_load(%dst: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 1]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_write> %dst, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  lsir.wait %token : !lsir.load_token
  return %res : !amdgcn.vgpr_range<[? + 1]>
}

func.func @test_wait_single_store(%data: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  lsir.wait %token : !lsir.store_token
  return
}

func.func @test_wait_multiple_loads(%dst0: !amdgcn.vgpr_range<[? + 1]>, %dst1: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) -> (!amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 1]>) {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %res0, %token0 = lsir.load #amdgcn.addr_space<global, read_write> %dst0, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  %res1, %token1 = lsir.load #amdgcn.addr_space<global, read_write> %dst1, %addr, %c0, %c4 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  lsir.wait %token0, %token1 : !lsir.load_token, !lsir.load_token
  return %res0, %res1 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 1]>
}

func.func @test_wait_mixed_tokens(%dst: !amdgcn.vgpr_range<[? + 1]>, %data: !amdgcn.vgpr_range<[? + 1]>, %load_addr: !amdgcn.vgpr_range<[? + 2]>, %store_addr: !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 1]> {
  %c0 = arith.constant 0 : i32
  %res, %load_token = lsir.load #amdgcn.addr_space<global, read_write> %dst, %load_addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  %store_token = lsir.store #amdgcn.addr_space<global, read_write> %data, %store_addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  lsir.wait %load_token, %store_token : !lsir.load_token, !lsir.store_token
  return %res : !amdgcn.vgpr_range<[? + 1]>
}

func.func @test_load_with_dependencies(%dst: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>, %data: !amdgcn.vgpr_range<[? + 1]>) -> !amdgcn.vgpr_range<[? + 1]> {
  %c0 = arith.constant 0 : i32
  %store_token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  %res, %load_token = lsir.load #amdgcn.addr_space<global, read_write> %dst, %addr, %c0, %c0 dependencies %store_token : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32, !lsir.store_token
  return %res : !amdgcn.vgpr_range<[? + 1]>
}

func.func @test_store_with_dependencies(%data0: !amdgcn.vgpr_range<[? + 1]>, %data1: !amdgcn.vgpr_range<[? + 1]>, %addr: !amdgcn.vgpr_range<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %token0 = lsir.store #amdgcn.addr_space<global, read_write> %data0, %addr, %c0, %c0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32
  %token1 = lsir.store #amdgcn.addr_space<global, read_write> %data1, %addr, %c0, %c4 dependencies %token0 : !amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>, i32, i32, !lsir.store_token
  lsir.wait %token1 : !lsir.store_token
  return
}

func.func @test_select_reg_condition(%dst: !amdgcn.vgpr, %cond: !amdgcn.vgpr, %tv: !amdgcn.vgpr, %fv: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.select %dst, %cond, %tv, %fv : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_select_i1_condition(%dst: !amdgcn.sgpr, %cond: i1, %tv: !amdgcn.sgpr, %fv: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %0 = lsir.select %dst, %cond, %tv, %fv : !amdgcn.sgpr, i1, !amdgcn.sgpr, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

func.func @test_select_i1_imm_operands(%dst: !amdgcn.sgpr, %cond: i1) -> !amdgcn.sgpr {
  %c42 = arith.constant 42 : i32
  %c99 = arith.constant 99 : i32
  %0 = lsir.select %dst, %cond, %c42, %c99 : !amdgcn.sgpr, i1, i32, i32
  return %0 : !amdgcn.sgpr
}
