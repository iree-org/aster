// RUN: aster-opt %s --verify-roundtrip
amdgcn.module @by_val_store_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @by_val_store arguments <[
    #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %val = amdgcn.load_arg 0 : !amdgcn.sgpr
    %out_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %v_a = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v_a, %val
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr

    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_a ins %c2, %tid_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    %tok = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
