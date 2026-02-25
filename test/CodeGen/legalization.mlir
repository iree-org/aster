// RUN: aster-opt --aster-legalizer --canonicalize --cse %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1, s2] -> (s2 * (s0 * 8 + s1))>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s3 * (s1 + s0 * s2))>

// CHECK-LABEL:   func.func @test_dim_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> index {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           return %[[CONSTANT_0]] : index
// CHECK:         }
func.func @test_dim_static(%m : memref<4x8xf32, #ptr.generic_space>) -> index {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %m, %c0 : memref<4x8xf32, #ptr.generic_space>
  return %dim : index
}

// CHECK-LABEL:   func.func @test_dim_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> index {
// CHECK:           return %[[ARG1]] : index
// CHECK:         }
func.func @test_dim_dynamic(%m : memref<4x?xf32, #ptr.generic_space>) -> index {
  %c1 = arith.constant 1 : index
  %dim = memref.dim %m, %c1 : memref<4x?xf32, #ptr.generic_space>
  return %dim : index
}

// CHECK-LABEL:   func.func @test_extract_metadata_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> (index, index, index, index, index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           return %[[CONSTANT_0]], %[[CONSTANT_1]], %[[CONSTANT_2]], %[[CONSTANT_2]], %[[CONSTANT_3]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_static(%m : memref<4x8xf32, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<4x8xf32, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_extract_metadata_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index, index, index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           return %[[CONSTANT_0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[CONSTANT_1]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_dynamic(%m : memref<?x?xf32, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<?x?xf32, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_extract_metadata_strided(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> (index, index, index, index, index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           return %[[ARG1]], %[[CONSTANT_0]], %[[CONSTANT_1]], %[[ARG2]], %[[CONSTANT_2]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_strided(%m : memref<4x8xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<4x8xf32, strided<[?, 1], offset: ?>, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_load_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> f32 {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG1]], %[[ARG2]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD_0:.*]] = ptr.load %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space> -> f32
// CHECK:           return %[[LOAD_0]] : f32
// CHECK:         }
func.func @test_load_static(%m : memref<4x8xf32, #ptr.generic_space>, %i: index, %j: index) -> f32 {
  %r = memref.load %m[%i, %j] : memref<4x8xf32, #ptr.generic_space>
  return %r : f32
}

// CHECK-LABEL:   func.func @test_load_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index) -> f32 {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[ARG4]], %[[ARG5]], %[[ARG3]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD_0:.*]] = ptr.load %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space> -> f32
// CHECK:           return %[[LOAD_0]] : f32
// CHECK:         }
func.func @test_load_dynamic(%m : memref<?x?xf32, #ptr.generic_space>, %i: index, %j: index) -> f32 {
  %r = memref.load %m[%i, %j] : memref<?x?xf32, #ptr.generic_space>
  return %r : f32
}

// CHECK-LABEL:   func.func @test_store_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR_ADD_0]] : f32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @test_store_static(%m : memref<4x8xf32, #ptr.generic_space>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #ptr.generic_space>
  return
}

// CHECK-LABEL:   func.func @test_store_default_space(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR_ADD_0]] : f32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @test_store_default_space(%m : memref<4x8xf32>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32>
  return
}

// CHECK-LABEL:   func.func @test_store_gpu_global(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#amdgcn.addr_space<global, read_write>>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR_ADD_0]] : f32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
// CHECK:           return
// CHECK:         }
func.func @test_store_gpu_global(%m : memref<4x8xf32, #gpu.address_space<global>>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #gpu.address_space<global>>
  return
}

// CHECK-LABEL:   func.func @test_store_gpu_shared(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#amdgcn.addr_space<local, read_write>>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TYPE_OFFSET_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#amdgcn.addr_space<local, read_write>>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR_ADD_0]] : f32, !ptr.ptr<#amdgcn.addr_space<local, read_write>>
// CHECK:           return
// CHECK:         }
func.func @test_store_gpu_shared(%m : memref<4x8xf32, #gpu.address_space<workgroup>>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #gpu.address_space<workgroup>>
  return
}
