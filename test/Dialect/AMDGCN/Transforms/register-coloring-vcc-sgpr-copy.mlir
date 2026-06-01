// RUN: aster-opt %s --amdgcn-register-coloring --amdgcn-post-reg-alloc-legalization | FileCheck %s

// CHECK-LABEL: kernel @vcc_sgpr_copy
// CHECK:         s_mov_b64 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.sgpr<[0 : 2]>) ins(!amdgcn.vcc<0>)
// CHECK-NOT:     lsir.copy
amdgcn.module @bug62 target = #amdgcn.target<gfx942> {
  amdgcn.kernel @vcc_sgpr_copy {
    %va     = amdgcn.alloca : !amdgcn.vgpr<?>
    %vcc_lo = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %vcc_hi = amdgcn.alloca : !amdgcn.vcc_hi<0>
    %vcc    = amdgcn.make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    %sp0    = amdgcn.alloca : !amdgcn.sgpr<?>
    %sp1    = amdgcn.alloca : !amdgcn.sgpr<?>
    %spill  = amdgcn.make_register_range %sp0, %sp1 : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
    %c60    = arith.constant 60 : i32
    amdgcn.v_cmp_le_u32 outs(%vcc) ins(%c60, %va) : outs(!amdgcn.vcc<0>) ins(i32, !amdgcn.vgpr<?>)
    lsir.copy %spill, %vcc : !amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vcc<0>
    test_inst ins %spill : (!amdgcn.sgpr<[? : ? + 2]>) -> ()
    amdgcn.end_kernel
  }
}
