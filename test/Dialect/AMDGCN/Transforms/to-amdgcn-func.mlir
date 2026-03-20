// RUN: aster-opt %s --aster-legalizer --aster-amdgcn-set-abi | FileCheck %s

// CHECK-LABEL: func.func @memref_kernel_to_gpu_kernel(
// CHECK-SAME:    %[[ARG0:.*]]: !ptr.ptr<#amdgcn.addr_space<global, read_write>>,
// CHECK-SAME:    %[[ARG1:.*]]: !ptr.ptr<#amdgcn.addr_space<global, read_write>>
// CHECK-SAME:    abi = (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) -> ()
// CHECK-SAME:    gpu.host_abi = {align = array<i32: 8, 8>, size = array<i32: 8, 8>
// CHECK-SAME:    gpu.kernel

// CHECK-LABEL: func.func private @plain_helper(
// CHECK-NOT:     gpu.kernel
// CHECK-NOT:     gpu.host_abi
module attributes {dlti.dl_spec = #dlti.dl_spec<
  !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>,
  !ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>>} {
  amdgcn.module @m target = <gfx942> isa = <cdna3> {
    func.func @memref_kernel_to_gpu_kernel(
      %a: memref<4xf32>,
      %b: memref<4xf32>
    ) attributes {gpu.kernel} {
      return
    }
    func.func private @plain_helper(%a: memref<4xf32>) -> memref<4xf32> {
      return %a : memref<4xf32>
    }
  }
}
