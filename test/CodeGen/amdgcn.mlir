// RUN: aster-opt %s --aster-codegen --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func private @test_mfma_f32_16x16x16_f16(
// CHECK-SAME:      %[[A:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[B:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[C:.*]]: !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]> {
// CHECK:           %[[RESULT:.*]] = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[C]], %[[A]], %[[B]], %[[C]] : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
// CHECK:           return %[[RESULT]] : !amdgcn.vgpr_range<[? + 4]>
// CHECK:         }
func.func private @test_mfma_f32_16x16x16_f16(%a: vector<4xf16>, %b: vector<4xf16>, %c: vector<4xf32>) -> vector<4xf32>
    attributes {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} {
  %result = amdgpu.mfma 16x16x16 %a * %b + %c blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  return {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} %result : vector<4xf32>
}

// CHECK-LABEL:   func.func private @test_mfma_f32_16x16x16_bf16(
// CHECK-SAME:      %[[A:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[B:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[C:.*]]: !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]> {
// CHECK:           %[[RESULT:.*]] = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_bf16> %[[C]], %[[A]], %[[B]], %[[C]] : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
// CHECK:           return %[[RESULT]] : !amdgcn.vgpr_range<[? + 4]>
// CHECK:         }
func.func private @test_mfma_f32_16x16x16_bf16(%a: vector<4xbf16>, %b: vector<4xbf16>, %c: vector<4xf32>) -> vector<4xf32>
    attributes {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} {
  %result = amdgpu.mfma 16x16x16 %a * %b + %c blgp = none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
  return {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} %result : vector<4xf32>
}

// CHECK-LABEL:   func.func private @test_mfma_f32_16x16x16_f16_attrs(
// CHECK-SAME:      %[[A:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[B:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[C:.*]]: !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]> {
// CHECK:           %[[RESULT:.*]] = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[C]], %[[A]], %[[B]], %[[C]] cbsz = 1 abid = 1 blgp = 2 : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]> -> !amdgcn.vgpr_range<[? + 4]>
// CHECK:           return %[[RESULT]] : !amdgcn.vgpr_range<[? + 4]>
// CHECK:         }
func.func private @test_mfma_f32_16x16x16_f16_attrs(%a: vector<4xf16>, %b: vector<4xf16>, %c: vector<4xf32>) -> vector<4xf32>
    attributes {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} {
  %result = amdgpu.mfma 16x16x16 %a * %b + %c {cbsz = 1 : i32, abid = 1 : i32, blocks = 1 : i32} blgp = bcast_second_32 : vector<4xf16>, vector<4xf16>, vector<4xf32>
  return {abi = (!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr_range<[? + 4]>) -> !amdgcn.vgpr_range<[? + 4]>} %result : vector<4xf32>
}
