// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

amdgcn.module @vop3 target = <gfx942> {
  // CHECK-LABEL: .globl v_lshl_add_u64
  kernel @v_lshl_add_u64 {
    %0 = alloca : !amdgcn.vgpr<0>
    %1 = alloca : !amdgcn.vgpr<1>
    %2 = alloca : !amdgcn.sgpr<0>
    %3 = alloca : !amdgcn.sgpr<1>
    %4 = alloca : !amdgcn.vgpr<2>
    %5 = alloca : !amdgcn.vgpr<3>
    %6 = make_register_range %0, %1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %7 = make_register_range %2, %3 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %8 = make_register_range %4, %5 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %c0_i32 = arith.constant 0 : i32
    // CHECK: v_lshl_add_u64 v[0:1], v[2:3], 0, s[0:1]
    amdgcn.v_lshl_add_u64 outs(%6) ins(%8, %c0_i32, %7) : outs(!amdgcn.vgpr<[0 : 2]>) ins(!amdgcn.vgpr<[2 : 4]>, i32, !amdgcn.sgpr<[0 : 2]>)
    end_kernel
  }
}
