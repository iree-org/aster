// RUN: aster-opt %s --verify-roundtrip
amdgcn.module @by_val_store_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @by_val_store arguments <[
    #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %val = amdgcn.load_arg 0 : !amdgcn.sgpr
    %out_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0

    %v_a = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.v_mov_b32 outs(%v_a) ins(%val) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %tok = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
