// RUN: aster-opt %s --amdgcn-mem2reg | FileCheck %s

// CHECK-LABEL:   func.func @test_struct_promotion(
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.store
// CHECK-NOT:     memref.load
// CHECK:         return %{{.*}} : !aster_utils.struct<i: index, j: index>
// CHECK:       }
func.func @test_struct_promotion(%0: !aster_utils.struct<i: index, j: index>) -> !aster_utils.struct<i: index, j: index> {
  %c0 = arith.constant 0 : index
  %struct_memref = memref.alloca() : memref<1x!aster_utils.struct<i: index, j: index>>
  memref.store %0, %struct_memref[%c0] : memref<1x!aster_utils.struct<i: index, j: index>>
  %loaded_struct = memref.load %struct_memref[%c0] : memref<1x!aster_utils.struct<i: index, j: index>>
  return %loaded_struct : !aster_utils.struct<i: index, j: index>
}

// CHECK-LABEL:   func.func @test_future_struct_promotion
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.store
// CHECK-NOT:     memref.load
// CHECK:         return {{.*}} : !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>
func.func @test_future_struct_promotion(%arg0: !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>) -> !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>> {
  %c0 = arith.constant 0 : index
  %future_memref = memref.alloca() : memref<1x!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
  memref.store %arg0, %future_memref[%c0] : memref<1x!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
  %loaded_future = memref.load %future_memref[%c0] : memref<1x!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
  return %loaded_future : !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>
}
