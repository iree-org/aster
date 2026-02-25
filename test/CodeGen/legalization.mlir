// RUN: aster-opt --aster-legalizer --canonicalize --cse %s | FileCheck %s

// CHECK: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s2 * (s0 * 4 + s1))>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s2 * (s0 * 8 + s1))>
// CHECK: #[[$MAP3:.+]] = affine_map<()[s0, s1, s2, s3] -> (s3 * (s2 + s0 * s1))>

// Verify that the legalizer attaches AMDGCN pointer size data layout.
// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec
// CHECK-SAME: addr_space<local
// CHECK-SAME: size = 32, abi = 32, preferred = 32
// CHECK-SAME: addr_space<global
// CHECK-SAME: size = 64, abi = 64, preferred = 64

// -----
// Vector legalization tests
// -----
// CHECK-LABEL:   func.func @element_wise(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x2xf32>, %[[ARG1:.*]]: vector<2x2xf32>, %[[ARG2:.*]]: vector<2x2xf32>) -> vector<2x2xf32> {
// CHECK:           %[[TO0:.*]]:4 = vector.to_elements %[[ARG0]] : vector<2x2xf32>
// CHECK:           %[[TO1:.*]]:4 = vector.to_elements %[[ARG1]] : vector<2x2xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[TO0]]#0, %[[TO1]]#0 : f32
// CHECK:           %[[ADD1:.*]] = arith.addf %[[TO0]]#1, %[[TO1]]#1 : f32
// CHECK:           %[[ADD2:.*]] = arith.addf %[[TO0]]#2, %[[TO1]]#2 : f32
// CHECK:           %[[ADD3:.*]] = arith.addf %[[TO0]]#3, %[[TO1]]#3 : f32
// CHECK:           %[[TO2:.*]]:4 = vector.to_elements %[[ARG2]] : vector<2x2xf32>
// CHECK:           %[[ADD4:.*]] = arith.addf %[[ADD0]], %[[TO2]]#0 : f32
// CHECK:           %[[ADD5:.*]] = arith.addf %[[ADD1]], %[[TO2]]#1 : f32
// CHECK:           %[[ADD6:.*]] = arith.addf %[[ADD2]], %[[TO2]]#2 : f32
// CHECK:           %[[ADD7:.*]] = arith.addf %[[ADD3]], %[[TO2]]#3 : f32
// CHECK:           %[[FROM:.*]] = vector.from_elements %[[ADD4]], %[[ADD5]], %[[ADD6]], %[[ADD7]] : vector<2x2xf32>
// CHECK:           return %[[FROM]] : vector<2x2xf32>
// CHECK:         }
func.func @element_wise(%lhs: vector<2x2xf32>, %rhs: vector<2x2xf32>, %rhs2: vector<2x2xf32>) -> vector<2x2xf32> {
  %result = arith.addf %lhs, %rhs : vector<2x2xf32>
  %r1 = arith.addf %result, %rhs2 : vector<2x2xf32>
  return %r1 : vector<2x2xf32>
}

// CHECK-LABEL:   func.func @load(
// CHECK-SAME:                    %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> vector<2x2xf32> {
// CHECK:           %[[LOAD0:.*]] = ptr.load %[[ARG0]] : !ptr.ptr<#ptr.generic_space> -> vector<2xf32>
// CHECK:           %[[TO0:.*]]:2 = vector.to_elements %[[LOAD0]] : vector<2xf32>
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP0]](){{\[}}%[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD1:.*]] = ptr.load %[[PTR]] : !ptr.ptr<#ptr.generic_space> -> vector<2xf32>
// CHECK:           %[[TO1:.*]]:2 = vector.to_elements %[[LOAD1]] : vector<2xf32>
// CHECK:           %[[FROM:.*]] = vector.from_elements %[[TO0]]#0, %[[TO0]]#1, %[[TO1]]#0, %[[TO1]]#1 : vector<2x2xf32>
// CHECK:           return %[[FROM]] : vector<2x2xf32>
// CHECK:         }
func.func @load(%m : memref<4x4xf32, #ptr.generic_space>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %v = vector.load %m[%c0, %c1] : memref<4x4xf32, #ptr.generic_space>, vector<2x2xf32>
  return %v : vector<2x2xf32>
}

// CHECK-LABEL:   func.func @store(
// CHECK-SAME:                     %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: vector<2x2xf32>) {
// CHECK:           %[[TO:.*]]:4 = vector.to_elements %[[ARG1]] : vector<2x2xf32>
// CHECK:           %[[FROM0:.*]] = vector.from_elements %[[TO]]#0, %[[TO]]#1 : vector<2xf32>
// CHECK:           ptr.store %[[FROM0]], %[[ARG0]] : vector<2xf32>, !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[FROM1:.*]] = vector.from_elements %[[TO]]#2, %[[TO]]#3 : vector<2xf32>
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP0]](){{\[}}%[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           ptr.store %[[FROM1]], %[[PTR]] : vector<2xf32>, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @store(%m : memref<4x4xf32, #ptr.generic_space>, %v: vector<2x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  vector.store %v, %m[%c0, %c1] : memref<4x4xf32, #ptr.generic_space>, vector<2x2xf32>
  return
}

// CHECK-LABEL:   func.func @load_comp_store(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: vector<2x2xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY0:.*]] = affine.apply #[[$MAP1]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR0:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD0:.*]] = ptr.load %[[PTR0]] : !ptr.ptr<#ptr.generic_space> -> vector<2xf32>
// CHECK:           %[[TO0:.*]]:2 = vector.to_elements %[[LOAD0]] : vector<2xf32>
// CHECK:           %[[ADDI:.*]] = arith.addi %[[ARG2]], %[[C1]] overflow<nsw> : index
// CHECK:           %[[APPLY1:.*]] = affine.apply #[[$MAP1]](){{\[}}%[[ADDI]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR1:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY1]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD1:.*]] = ptr.load %[[PTR1]] : !ptr.ptr<#ptr.generic_space> -> vector<2xf32>
// CHECK:           %[[TO1:.*]]:2 = vector.to_elements %[[LOAD1]] : vector<2xf32>
// CHECK:           %[[TO2:.*]]:4 = vector.to_elements %[[ARG1]] : vector<2x2xf32>
// CHECK:           %[[ADD0:.*]] = arith.addf %[[TO0]]#0, %[[TO2]]#0 : f32
// CHECK:           %[[ADD1:.*]] = arith.addf %[[TO0]]#1, %[[TO2]]#1 : f32
// CHECK:           %[[ADD2:.*]] = arith.addf %[[TO1]]#0, %[[TO2]]#2 : f32
// CHECK:           %[[ADD3:.*]] = arith.addf %[[TO1]]#1, %[[TO2]]#3 : f32
// CHECK:           %[[FROM0:.*]] = vector.from_elements %[[ADD0]], %[[ADD1]] : vector<2xf32>
// CHECK:           ptr.store %[[FROM0]], %[[PTR0]] : vector<2xf32>, !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[FROM1:.*]] = vector.from_elements %[[ADD2]], %[[ADD3]] : vector<2xf32>
// CHECK:           ptr.store %[[FROM1]], %[[PTR1]] : vector<2xf32>, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @load_comp_store(%m : memref<4x4xf32, #ptr.generic_space>, %in: vector<2x2xf32>, %i: index, %j: index) {
  %v = vector.load %m[%i, %j] : memref<4x4xf32, #ptr.generic_space>, vector<2x2xf32>
  %r1 = arith.addf %v, %in : vector<2x2xf32>
  vector.store %r1, %m[%i, %j] : memref<4x4xf32, #ptr.generic_space>, vector<2x2xf32>
  return
}

// CHECK-LABEL:   func.func @extract_scalar(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x3xf32>) -> f32 {
// CHECK:           %[[TO:.*]]:6 = vector.to_elements %[[ARG0]] : vector<2x3xf32>
// CHECK:           return %[[TO]]#5 : f32
// CHECK:         }
func.func @extract_scalar(%v: vector<2x3xf32>) -> f32 {
  %r = vector.extract %v[1, 2] : f32 from vector<2x3xf32>
  return %r : f32
}

// CHECK-LABEL:   func.func @extract_subvector(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x3xf32>) -> vector<3xf32> {
// CHECK:           %[[TO:.*]]:6 = vector.to_elements %[[ARG0]] : vector<2x3xf32>
// CHECK:           %[[FROM:.*]] = vector.from_elements %[[TO]]#0, %[[TO]]#1, %[[TO]]#2 : vector<3xf32>
// CHECK:           return %[[FROM]] : vector<3xf32>
// CHECK:         }
func.func @extract_subvector(%v: vector<2x3xf32>) -> vector<3xf32> {
  %r = vector.extract %v[0] : vector<3xf32> from vector<2x3xf32>
  return %r : vector<3xf32>
}

// CHECK-LABEL:   func.func @insert_scalar(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x3xf32>, %[[ARG1:.*]]: f32) -> vector<2x3xf32> {
// CHECK:           %[[TO:.*]]:6 = vector.to_elements %[[ARG0]] : vector<2x3xf32>
// CHECK:           %[[FROM:.*]] = vector.from_elements %[[TO]]#0, %[[ARG1]], %[[TO]]#2, %[[TO]]#3, %[[TO]]#4, %[[TO]]#5 : vector<2x3xf32>
// CHECK:           return %[[FROM]] : vector<2x3xf32>
// CHECK:         }
func.func @insert_scalar(%v: vector<2x3xf32>, %s: f32) -> vector<2x3xf32> {
  %r = vector.insert %s, %v[0, 1] : f32 into vector<2x3xf32>
  return %r : vector<2x3xf32>
}

// CHECK-LABEL:   func.func @insert_subvector(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x3xf32>, %[[ARG1:.*]]: vector<3xf32>) -> vector<2x3xf32> {
// CHECK:           %[[TO0:.*]]:6 = vector.to_elements %[[ARG0]] : vector<2x3xf32>
// CHECK:           %[[TO1:.*]]:3 = vector.to_elements %[[ARG1]] : vector<3xf32>
// CHECK:           %[[FROM:.*]] = vector.from_elements %[[TO0]]#0, %[[TO0]]#1, %[[TO0]]#2, %[[TO1]]#0, %[[TO1]]#1, %[[TO1]]#2 : vector<2x3xf32>
// CHECK:           return %[[FROM]] : vector<2x3xf32>
// CHECK:         }
func.func @insert_subvector(%v: vector<2x3xf32>, %sub: vector<3xf32>) -> vector<2x3xf32> {
  %r = vector.insert %sub, %v[1] : vector<3xf32> into vector<2x3xf32>
  return %r : vector<2x3xf32>
}

// CHECK-LABEL:   func.func @broadcast_scalar(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> vector<2x3xf32> {
// CHECK:           %[[BC:.*]] = vector.broadcast %[[ARG0]] : f32 to vector<2x3xf32>
// CHECK:           return %[[BC]] : vector<2x3xf32>
// CHECK:         }
func.func @broadcast_scalar(%s: f32) -> vector<2x3xf32> {
  %r = vector.broadcast %s : f32 to vector<2x3xf32>
  return %r : vector<2x3xf32>
}

// -----
// Memref legalization tests
// -----

// CHECK-LABEL:   func.func @test_dim_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> index {
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           return %[[C4]] : index
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
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           return %[[C0]], %[[C4]], %[[C8]], %[[C8]], %[[C1]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_static(%m : memref<4x8xf32, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<4x8xf32, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_extract_metadata_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> (index, index, index, index, index) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           return %[[C0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[C1]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_dynamic(%m : memref<?x?xf32, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<?x?xf32, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_extract_metadata_strided(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> (index, index, index, index, index) {
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           return %[[ARG1]], %[[C4]], %[[C8]], %[[ARG2]], %[[C1]] : index, index, index, index, index
// CHECK:         }
func.func @test_extract_metadata_strided(%m : memref<4x8xf32, strided<[?, 1], offset: ?>, #ptr.generic_space>) -> (index, index, index, index, index) {
  %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %m
      : memref<4x8xf32, strided<[?, 1], offset: ?>, #ptr.generic_space> -> memref<f32, #ptr.generic_space>, index, index, index, index, index
  return %offset, %size0, %size1, %stride0, %stride1 : index, index, index, index, index
}

// CHECK-LABEL:   func.func @test_load_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> f32 {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP2]](){{\[}}%[[ARG1]], %[[ARG2]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD:.*]] = ptr.load %[[PTR]] : !ptr.ptr<#ptr.generic_space> -> f32
// CHECK:           return %[[LOAD]] : f32
// CHECK:         }
func.func @test_load_static(%m : memref<4x8xf32, #ptr.generic_space>, %i: index, %j: index) -> f32 {
  %r = memref.load %m[%i, %j] : memref<4x8xf32, #ptr.generic_space>
  return %r : f32
}

// CHECK-LABEL:   func.func @test_load_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index) -> f32 {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP3]](){{\[}}%[[ARG4]], %[[ARG3]], %[[ARG5]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[LOAD:.*]] = ptr.load %[[PTR]] : !ptr.ptr<#ptr.generic_space> -> f32
// CHECK:           return %[[LOAD]] : f32
// CHECK:         }
func.func @test_load_dynamic(%m : memref<?x?xf32, #ptr.generic_space>, %i: index, %j: index) -> f32 {
  %r = memref.load %m[%i, %j] : memref<?x?xf32, #ptr.generic_space>
  return %r : f32
}

// CHECK-LABEL:   func.func @test_store_static(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP2]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR]] : f32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @test_store_static(%m : memref<4x8xf32, #ptr.generic_space>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #ptr.generic_space>
  return
}

// CHECK-LABEL:   func.func @test_store_default_space(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP2]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR]] : f32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
func.func @test_store_default_space(%m : memref<4x8xf32>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32>
  return
}

// CHECK-LABEL:   func.func @test_store_gpu_global(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#amdgcn.addr_space<global, read_write>>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP2]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#amdgcn.addr_space<global, read_write>>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR]] : f32, !ptr.ptr<#amdgcn.addr_space<global, read_write>>
// CHECK:           return
// CHECK:         }
func.func @test_store_gpu_global(%m : memref<4x8xf32, #gpu.address_space<global>>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #gpu.address_space<global>>
  return
}

// CHECK-LABEL:   func.func @test_store_gpu_shared(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#amdgcn.addr_space<local, read_write>>, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) {
// CHECK:           %[[TOFF:.*]] = ptr.type_offset f32 : index
// CHECK:           %[[APPLY:.*]] = affine.apply #[[$MAP2]](){{\[}}%[[ARG2]], %[[ARG3]], %[[TOFF]]]
// CHECK:           %[[PTR:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY]] : !ptr.ptr<#amdgcn.addr_space<local, read_write>>, index
// CHECK:           ptr.store %[[ARG1]], %[[PTR]] : f32, !ptr.ptr<#amdgcn.addr_space<local, read_write>>
// CHECK:           return
// CHECK:         }
func.func @test_store_gpu_shared(%m : memref<4x8xf32, #gpu.address_space<workgroup>>, %v: f32, %i: index, %j: index) {
  memref.store %v, %m[%i, %j] : memref<4x8xf32, #gpu.address_space<workgroup>>
  return
}
