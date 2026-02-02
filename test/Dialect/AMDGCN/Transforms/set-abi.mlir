// RUN: aster-opt %s --aster-amdgcn-set-abi | FileCheck %s

// CHECK-LABEL:   func.func private @test_add_i16(
// CHECK-SAME:      %[[ARG0:.*]]: i16, %[[ARG1:.*]]: i16) -> i16 attributes {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i16
// CHECK:           return {abi = (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr} %[[ADDI_0]] : i16
// CHECK:         }

func.func @test_add_i16(%arg0: i16, %arg1: i16) -> i16 {
  %0 = arith.addi %arg0, %arg1 : i16
  return %0 : i16
}

// CHECK-LABEL:   func.func private @test_inter(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32 attributes {abi = (!amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.sgpr} {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK:           return {abi = (!amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.sgpr} %[[ADDI_0]] : i32
// CHECK:         }
func.func private @test_inter(%0: i32, %1: i32) -> i32 {
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:      %[[ARG0:.*]]: i32) attributes {abi = (!amdgcn.sgpr) -> (), gpu.host_abi = {align = array<i32: 4>, size = array<i32: 4>, type = (i32) -> ()}, gpu.kernel} {
// CHECK:           %[[VAL_0:.*]] = call @test_inter(%[[ARG0]], %[[ARG0]]) {abi = (!amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.sgpr} : (i32, i32) -> i32
// CHECK:           return
// CHECK:         }
func.func @test_kernel(%0: i32) attributes{gpu.kernel} {
  %1 = func.call @test_inter(%0, %0) : (i32, i32) -> i32
  return
}
