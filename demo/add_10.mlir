module {
  amdgcn.module @add_10_module target = <gfx942> isa = <cdna3> {
    kernel @kernel {
      %0 = alloca : !amdgcn.vgpr<10>
      %1 = alloca : !amdgcn.vgpr<11>
      %2 = alloca : !amdgcn.vgpr<12>
      %c1_i32 = arith.constant 1 : i32
      amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %c1_i32 : (!amdgcn.vgpr<11>, i32) -> ()
      %c2_i32 = arith.constant 2 : i32
      amdgcn.vop1.vop1 <v_mov_b32_e32> %2, %c2_i32 : (!amdgcn.vgpr<12>, i32) -> ()
      vop2 v_add_u32 outs %0 ins %1, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      vop2 v_add_u32 outs %0 ins %0, %2 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      end_kernel
    }
  }
}
