// RUN: aster-opt %s --sroa --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @test_sroa
//       CHECK:     memref.alloca() : memref<!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
//   CHECK-NOT:     memref.alloca
func.func @test_sroa() -> !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>> {
  %c0 = arith.constant 0 : index
  %future_memref = memref.alloca() : memref<4x!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
  %loaded_future = memref.load %future_memref[%c0] : memref<4x!aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>>
  return %loaded_future : !aster_utils.struct<value: !aster_utils.any, foo: !aster_utils.struct<value: !aster_utils.any>, token: !amdgcn.read_token<flat>>
}
