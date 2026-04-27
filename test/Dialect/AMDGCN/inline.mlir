// RUN: aster-opt %s --inline | FileCheck %s

//===----------------------------------------------------------------------===//
// Test inline pass
//===----------------------------------------------------------------------===//

func.func @helper(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> ()
  return %0 : !amdgcn.vgpr<1>
}

// CHECK-LABEL: func.func @main
//       CHECK:   amdgcn.alloca
//   CHECK-NOT:   call
func.func @main(%arg: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %result = call @helper(%arg) : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %result : !amdgcn.vgpr<1>
}

//===----------------------------------------------------------------------===//
// Test inline of functions containing ptr dialect ops (ptr.ptr_add).
//===----------------------------------------------------------------------===//

!gptr = !ptr.ptr<#ptr.generic_space>
!vx2 = !amdgcn.vgpr<[? + 2]>

func.func @ptr_helper(%base: !gptr, %off: i32) -> !vx2 {
  %addr = ptr.ptr_add %base, %off : !gptr, i32
  %reg = lsir.to_reg %addr : !gptr -> !vx2
  return %reg : !vx2
}

// CHECK-LABEL: func.func @ptr_caller
//       CHECK:   ptr.ptr_add
//       CHECK:   lsir.to_reg
//   CHECK-NOT:   call @ptr_helper
func.func @ptr_caller(%base: !gptr, %off: i32) -> !vx2 {
  %result = call @ptr_helper(%base, %off) : (!gptr, i32) -> !vx2
  return %result : !vx2
}

//===----------------------------------------------------------------------===//
// Test inline pass within amdgcn.kernel
//===----------------------------------------------------------------------===//

module {
  amdgcn.module @kernel_module target = #amdgcn.target<gfx942> {
    func.func @kernel_helper(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
      %0 = amdgcn.alloca : !amdgcn.vgpr<1>
      amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> ()
      return %0 : !amdgcn.vgpr<1>
    }

    // CHECK-LABEL: kernel @kernel_main
    //   CHECK-NOT:   call
    amdgcn.kernel @kernel_main {
      %arg = amdgcn.alloca : !amdgcn.vgpr<0>
      %result = func.call @kernel_helper(%arg) : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
      amdgcn.end_kernel
    }
  }
}
