// RUN: aster-opt %s --inline | FileCheck %s

//===----------------------------------------------------------------------===//
// Test inline pass
//===----------------------------------------------------------------------===//

func.func @helper(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %1 : !amdgcn.vgpr<1>
}

// CHECK-LABEL: func.func @main
//       CHECK:   amdgcn.alloca
//       CHECK:   v_mov_b32_e32
//   CHECK-NOT:   call
func.func @main(%arg: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %result = call @helper(%arg) : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %result : !amdgcn.vgpr<1>
}

//===----------------------------------------------------------------------===//
// Test inline pass within amdgcn.kernel
//===----------------------------------------------------------------------===//

module {
  amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
    func.func @kernel_helper(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
      %0 = amdgcn.alloca : !amdgcn.vgpr<1>
      %1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
      return %1 : !amdgcn.vgpr<1>
    }

    // CHECK-LABEL: kernel @kernel_main
    //       CHECK:   v_mov_b32_e32
    //   CHECK-NOT:   call
    amdgcn.kernel @kernel_main {
      %arg = amdgcn.alloca : !amdgcn.vgpr<0>
      %result = func.call @kernel_helper(%arg) : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
      amdgcn.end_kernel
    }
  }
}
