module {
  amdgcn.module @add_10_module target = <gfx942> {
    kernel @kernel {
      %0 = alloca : !amdgcn.vgpr<10>
      %1 = alloca : !amdgcn.vgpr<11>
      %2 = alloca : !amdgcn.vgpr<12>
      %c1_i32 = arith.constant 1 : i32
      amdgcn.v_mov_b32 outs(%1) ins(%c1_i32) : outs(!amdgcn.vgpr<11>) ins(i32)
      %c2_i32 = arith.constant 2 : i32
      amdgcn.v_mov_b32 outs(%2) ins(%c2_i32) : outs(!amdgcn.vgpr<12>) ins(i32)
      amdgcn.v_add_u32 outs(%0) ins(%1, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<11>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%1, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<11>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      amdgcn.v_add_u32 outs(%0) ins(%0, %2) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.vgpr<10>, !amdgcn.vgpr<12>)
      end_kernel
    }
  }
}
