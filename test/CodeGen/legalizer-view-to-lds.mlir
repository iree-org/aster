// RUN: aster-opt %s --aster-legalizer | FileCheck %s

amdgcn.module @mod target = #amdgcn.target<gfx942> {
// CHECK-LABEL: func.func @test_view_to_lds
// CHECK:         amdgcn.alloc_lds 1024
// CHECK:         amdgcn.get_lds_offset
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.view
func.func @test_view_to_lds(%byte_shift: index, %val: f16) {
  %alloca = memref.alloca() : memref<1024xi8, 3>
  %view = memref.view %alloca[%byte_shift][] : memref<1024xi8, 3> to memref<16x32xf16, 3>
  %c0 = arith.constant 0 : index
  memref.store %val, %view[%c0, %c0] : memref<16x32xf16, 3>
  return
}

// CHECK-LABEL: func.func @test_view_default_space
// CHECK:         memref.alloca
// CHECK:         memref.view
func.func @test_view_default_space(%byte_shift: index, %val: f16) {
  %alloca = memref.alloca() : memref<1024xi8>
  %view = memref.view %alloca[%byte_shift][] : memref<1024xi8> to memref<16x32xf16>
  %c0 = arith.constant 0 : index
  memref.store %val, %view[%c0, %c0] : memref<16x32xf16>
  return
}
}
