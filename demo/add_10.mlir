module {
  amdgcn.module @add_10_module target = <gfx942> isa = <cdna3> {
    kernel @kernel {
      %0 = alloca : !amdgcn.vgpr<10>
      %1 = alloca : !amdgcn.vgpr<11>
      %2 = alloca : !amdgcn.vgpr<12>
      %c1_i32 = arith.constant 1 : i32
      %3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %c1_i32 : (!amdgcn.vgpr<11>, i32) -> !amdgcn.vgpr<11>
      %c2_i32 = arith.constant 2 : i32
      %4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %2, %c2_i32 : (!amdgcn.vgpr<12>, i32) -> !amdgcn.vgpr<12>
      %vdst0_res = vop2 v_add_u32 outs %0 ins %3, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>, !amdgcn.vgpr<12>
      %vdst0_res_0 = vop2 v_add_u32 outs %vdst0_res ins %vdst0_res, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_1 = vop2 v_add_u32 outs %vdst0_res_0 ins %vdst0_res_0, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_2 = vop2 v_add_u32 outs %vdst0_res_1 ins %vdst0_res_1, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_3 = vop2 v_add_u32 outs %vdst0_res_2 ins %vdst0_res_2, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_4 = vop2 v_add_u32 outs %vdst0_res_3 ins %vdst0_res_3, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_5 = vop2 v_add_u32 outs %vdst0_res_4 ins %vdst0_res_4, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_6 = vop2 v_add_u32 outs %vdst0_res_5 ins %vdst0_res_5, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_7 = vop2 v_add_u32 outs %vdst0_res_6 ins %vdst0_res_6, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      %vdst0_res_8 = vop2 v_add_u32 outs %vdst0_res_7 ins %vdst0_res_7, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<10>, !amdgcn.vgpr<12>
      end_kernel
    }
  }
}

