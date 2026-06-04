// Preexisting asm-level kernel.

amdgcn.module @pin_fusion_add4 target = #amdgcn.target<gfx942> {
  func.func private @add4(
      %a_ptr: !amdgcn.sgpr<[? + 2]>, %b_ptr: !amdgcn.sgpr<[? + 2]>)
      -> (!amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>) {
    %c0 = arith.constant 0 : i32
    %a0 = amdgcn.alloca : !amdgcn.vgpr<12>
    %a1 = amdgcn.alloca : !amdgcn.vgpr<13>
    %a2 = amdgcn.alloca : !amdgcn.vgpr<14>
    %a3 = amdgcn.alloca : !amdgcn.vgpr<15>
    %b0 = amdgcn.alloca : !amdgcn.vgpr<16>
    %b1 = amdgcn.alloca : !amdgcn.vgpr<17>
    %b2 = amdgcn.alloca : !amdgcn.vgpr<18>
    %b3 = amdgcn.alloca : !amdgcn.vgpr<19>
    %a = amdgcn.make_register_range %a0, %a1, %a2, %a3
        : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>
    %b = amdgcn.make_register_range %b0, %b1, %b2, %b3
        : !amdgcn.vgpr<16>, !amdgcn.vgpr<17>, !amdgcn.vgpr<18>, !amdgcn.vgpr<19>
    %off_a = amdgcn.alloca : !amdgcn.vgpr
    %off_b = amdgcn.alloca : !amdgcn.vgpr
    %off0_a = amdgcn.v_mov_b32 outs(%off_a) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %off0_b = amdgcn.v_mov_b32 outs(%off_b) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %ta = amdgcn.global_load_dwordx4 dest %a addr %a_ptr offset d(%off0_a) + c(%c0)
        : outs(!amdgcn.vgpr<[12 : 16]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
    %tb = amdgcn.global_load_dwordx4 dest %b addr %b_ptr offset d(%off0_b) + c(%c0)
        : outs(!amdgcn.vgpr<[16 : 20]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    %s0 = amdgcn.alloca : !amdgcn.vgpr<40>
    %s1 = amdgcn.alloca : !amdgcn.vgpr<41>
    %s2 = amdgcn.alloca : !amdgcn.vgpr<42>
    %s3 = amdgcn.alloca : !amdgcn.vgpr<43>
    amdgcn.v_add_f32 outs(%s0) ins(%a0, %b0) : outs(!amdgcn.vgpr<40>) ins(!amdgcn.vgpr<12>, !amdgcn.vgpr<16>)
    amdgcn.v_add_f32 outs(%s1) ins(%a1, %b1) : outs(!amdgcn.vgpr<41>) ins(!amdgcn.vgpr<13>, !amdgcn.vgpr<17>)
    amdgcn.v_add_f32 outs(%s2) ins(%a2, %b2) : outs(!amdgcn.vgpr<42>) ins(!amdgcn.vgpr<14>, !amdgcn.vgpr<18>)
    amdgcn.v_add_f32 outs(%s3) ins(%a3, %b3) : outs(!amdgcn.vgpr<43>) ins(!amdgcn.vgpr<15>, !amdgcn.vgpr<19>)
    return %s0, %s1, %s2, %s3 : !amdgcn.vgpr<40>, !amdgcn.vgpr<41>, !amdgcn.vgpr<42>, !amdgcn.vgpr<43>
  }
}
