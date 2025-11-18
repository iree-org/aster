// RUN: aster-opt %s --amdgcn-preload-library="library-paths=%p/preload-library-lib.mlir" | FileCheck %s

// CHECK-LABEL: amdgcn.module @test_module
// The library functions should be imported and replace the declarations.
// CHECK: func.func @wave_index(%arg0: index) -> index {
// CHECK:   %c64 = arith.constant 64 : index
// CHECK:   arith.divui %arg0, %c64 : index
// CHECK:   return
// CHECK: }
// CHECK: func.func @lane_index(%arg0: index) -> index {
// CHECK:   %c64 = arith.constant 64 : index
// CHECK:   arith.remui %arg0, %c64 : index
// CHECK:   return
// CHECK: }
// CHECK: func.func @test_kernel_func
amdgcn.module @test_module target = #amdgcn.target<gfx940> isa = #amdgcn.isa<cdna3> {
  // Declare library functions that will be imported.
  func.func private @wave_index(index) -> index
  func.func private @lane_index(index) -> index

  // This function calls the library functions.
  func.func @test_kernel_func(%tidx: index) -> (index, index) {
    %widx = func.call @wave_index(%tidx) : (index) -> index
    %lidx = func.call @lane_index(%tidx) : (index) -> index
    return %widx, %lidx : index, index
  }
}
