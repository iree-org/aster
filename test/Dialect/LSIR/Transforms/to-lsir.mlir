// RUN: aster-opt %s --aster-to-lsir --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func private @test_add(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[ADDI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_add(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.addi %arg0, %arg1 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_add_i16(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i16 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[ADDI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_add_i16(%arg0: i16, %arg1: i16) -> i16 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.addi %arg0, %arg1 : i16
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i16
}

// CHECK-LABEL:   func.func private @test_add_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 2], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i64 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 2], uniform = 0>, !amdgcn.ggpr<[? + 2], uniform = 0>, !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:           return %[[ADDI_0]] : !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:         }
func.func private @test_add_i64(%arg0: i64, %arg1: i64) -> i64 attributes {abi = (!amdgcn.ggpr<[? + 2], uniform = 0>, !amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0>} {
  %0 = arith.addi %arg0, %arg1 : i64
  return {abi = (!amdgcn.ggpr<[? + 2], uniform = 0>, !amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0>} %0 : i64
}

// CHECK-LABEL:   func.func private @test_add_chained(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG2:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ALLOCA_1:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDI_1:.*]] = lsir.addi i32 %[[ALLOCA_1]], %[[ADDI_0]], %[[ARG2]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[ADDI_1]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_add_chained(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.addi %0, %arg2 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %1 : i32
}

// CHECK-LABEL:   func.func private @test_add_constant(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i32 %[[ALLOCA_0]], %[[ARG0]], %[[CONSTANT_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, i32
// CHECK:           return %[[ADDI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_add_constant(%arg0: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %c42_i32 = arith.constant 42 : i32
  %0 = arith.addi %arg0, %c42_i32 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_thread_id() -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[THREAD_ID_0:.*]] = amdgcn.thread_id  x : !amdgcn.vgpr
// CHECK:           %[[REG_CAST_0:.*]] = lsir.reg_cast %[[THREAD_ID_0]] : !amdgcn.vgpr -> !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[REG_CAST_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_thread_id() -> i32 attributes {abi = () -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = aster_utils.thread_id  x
  return {abi = () -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_block_id() -> !amdgcn.ggpr<[? + 1], uniform = 1> {
// CHECK:           %[[BLOCK_ID_0:.*]] = amdgcn.block_id  x : !amdgcn.sgpr
// CHECK:           %[[REG_CAST_0:.*]] = lsir.reg_cast %[[BLOCK_ID_0]] : !amdgcn.sgpr -> !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           return %[[REG_CAST_0]] : !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:         }
func.func private @test_block_id() -> i32 attributes {abi = () -> !amdgcn.ggpr<[? + 1], uniform = 1>} {
  %0 = aster_utils.block_id  x
  return {abi = () -> !amdgcn.ggpr<[? + 1], uniform = 1>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_inter(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 1>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 1>) -> !amdgcn.ggpr<[? + 1], uniform = 1> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           return %[[ADDI_0]] : !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:         }
func.func private @test_inter(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>) -> !amdgcn.ggpr<[? + 1], uniform = 1>} {
  %0 = arith.addi %arg0, %arg1 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>) -> !amdgcn.ggpr<[? + 1], uniform = 1>} %0 : i32
}

// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr) attributes {gpu.kernel} {
// CHECK:           %[[REG_CAST_0:.*]] = lsir.reg_cast %[[ARG0]] : !amdgcn.sgpr -> !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           %[[REG_CAST_1:.*]] = lsir.reg_cast %[[ARG0]] : !amdgcn.sgpr -> !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           %[[VAL_0:.*]] = call @test_inter(%[[REG_CAST_0]], %[[REG_CAST_1]]) : (!amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>) -> !amdgcn.ggpr<[? + 1], uniform = 1>
// CHECK:           return
// CHECK:         }
func.func @test_kernel(%arg0: i32) attributes {abi = (!amdgcn.sgpr) -> (), gpu.kernel} {
  %0 = call @test_inter(%arg0, %arg0) {abi = (!amdgcn.ggpr<[? + 1], uniform = 1>, !amdgcn.ggpr<[? + 1], uniform = 1>) -> !amdgcn.ggpr<[? + 1], uniform = 1>} : (i32, i32) -> i32
  return
}

// CHECK-LABEL:   func.func private @test_constraint(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.vgpr
// CHECK:           %[[ADDI_0:.*]] = lsir.addi i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[REG_CAST_0:.*]] = lsir.reg_cast %[[ADDI_0]] : !amdgcn.vgpr -> !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[REG_CAST_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_constraint(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.addi %arg0, %arg1 : i32
  lsir.reg_constraint %0 {kind = #amdgcn.reg_kind<VGPR>} : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_xori(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[XORI_0:.*]] = lsir.xori i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[XORI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_xori(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.xori %arg0, %arg1 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_maxsi(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[MAXSI_0:.*]] = lsir.maxsi i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[MAXSI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_maxsi(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.maxsi %arg0, %arg1 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_maxui(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[MAXUI_0:.*]] = lsir.maxui i32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[MAXUI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_maxui(%arg0: i32, %arg1: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.maxui %arg0, %arg1 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_addf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[ADDF_0:.*]] = lsir.addf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[ADDF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_addf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.addf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_subf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[SUBF_0:.*]] = lsir.subf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[SUBF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_subf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.subf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_mulf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[MULF_0:.*]] = lsir.mulf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[MULF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_mulf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.mulf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_divf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[DIVF_0:.*]] = lsir.divf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[DIVF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_divf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.divf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_maximumf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[MAXIMUMF_0:.*]] = lsir.maximumf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[MAXIMUMF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_maximumf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.maximumf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_minimumf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[MINIMUMF_0:.*]] = lsir.minimumf f32 %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[MINIMUMF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_minimumf(%arg0: f32, %arg1: f32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.minimumf %arg0, %arg1 : f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_extsi(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[EXTSI_0:.*]] = lsir.extsi i32 from i16 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[EXTSI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_extsi(%arg0: i16) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.extsi %arg0 : i16 to i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_extui(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[EXTUI_0:.*]] = lsir.extui i32 from i16 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[EXTUI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_extui(%arg0: i16) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.extui %arg0 : i16 to i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_trunci(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[TRUNCI_0:.*]] = lsir.trunci i16 from i32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[TRUNCI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_trunci(%arg0: i32) -> i16 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.trunci %arg0 : i32 to i16
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i16
}

// CHECK-LABEL:   func.func private @test_extf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:           %[[EXTF_0:.*]] = lsir.extf f64 from f32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 2], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[EXTF_0]] : !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:         }
func.func private @test_extf(%arg0: f32) -> f64 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0>} {
  %0 = arith.extf %arg0 : f32 to f64
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 2], uniform = 0>} %0 : f64
}

// CHECK-LABEL:   func.func private @test_truncf(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[TRUNCF_0:.*]] = lsir.truncf f32 from f64 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 2], uniform = 0>
// CHECK:           return %[[TRUNCF_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_truncf(%arg0: f64) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.truncf %arg0 : f64 to f32
  return {abi = (!amdgcn.ggpr<[? + 2], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_fptosi(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[FPTOSI_0:.*]] = lsir.fptosi i32 from f32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[FPTOSI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_fptosi(%arg0: f32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.fptosi %arg0 : f32 to i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_fptoui(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[FPTOUI_0:.*]] = lsir.fptoui i32 from f32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[FPTOUI_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_fptoui(%arg0: f32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.fptoui %arg0 : f32 to i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}

// CHECK-LABEL:   func.func private @test_sitofp(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[SITOFP_0:.*]] = lsir.sitofp f32 from i32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[SITOFP_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_sitofp(%arg0: i32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.sitofp %arg0 : i32 to f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_uitofp(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[UITOFP_0:.*]] = lsir.uitofp f32 from i32 %[[ALLOCA_0]], %[[ARG0]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[UITOFP_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_uitofp(%arg0: i32) -> f32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.uitofp %arg0 : i32 to f32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : f32
}

// CHECK-LABEL:   func.func private @test_select(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG1:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>, %[[ARG2:.*]]: !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0> {
// CHECK:           %[[ALLOCA_0:.*]] = lsir.alloca : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           %[[SELECT_0:.*]] = lsir.select %[[ALLOCA_0]], %[[ARG0]], %[[ARG1]], %[[ARG2]] : !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:           return %[[SELECT_0]] : !amdgcn.ggpr<[? + 1], uniform = 0>
// CHECK:         }
func.func private @test_select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 attributes {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} {
  %0 = arith.select %arg0, %arg1, %arg2 : i32
  return {abi = (!amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>, !amdgcn.ggpr<[? + 1], uniform = 0>) -> !amdgcn.ggpr<[? + 1], uniform = 0>} %0 : i32
}
