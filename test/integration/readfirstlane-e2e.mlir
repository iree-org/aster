// RUN: aster-opt %s --verify-roundtrip

amdgcn.module @readfirstlane_mod target = #amdgcn.target<gfx942> {

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %out_ptr : !amdgcn.sgpr<[? + 2]>
  }

  // ----- Test 1: constant via readfirstlane -----
  // All lanes get 42 from VGPR -> readfirstlane -> SGPR -> broadcast -> store.
  // block=(64,1,1), grid=(1,1,1). Output: 64 dwords, all 42.
  amdgcn.kernel @readfirstlane_const arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr()
      : () -> !amdgcn.sgpr<[? + 2]>

    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    // Per-lane byte offset: tid_x * 4
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_a ins %c2, %tid
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    // Load constant 42 into a VGPR (all lanes get the same value).
    %v42_a = amdgcn.alloca : !amdgcn.vgpr
    %c42 = arith.constant 42 : i32
    %v42 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v42_a, %c42
      : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr

    // v_readfirstlane_b32: VGPR -> SGPR
    %s_dest = amdgcn.alloca : !amdgcn.sgpr
    %s_val = amdgcn.vop1.lane #amdgcn.inst<v_readfirstlane_b32> %s_dest, %v42
      : (!amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.sgpr

    // Broadcast SGPR back to all VGPR lanes.
    %v_dest = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v_dest, %s_val
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr

    // Store to global: out[tid_x] = 42
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 2: thread_id via readfirstlane (wave-uniform value) -----
  // tid_x of lane 0 is always 0 in a wavefront.  Use readfirstlane on tid_x
  // to capture that value into an SGPR, broadcast to all lanes, and store.
  // block=(64,1,1), grid=(1,1,1). Output: 64 dwords, all 0.
  amdgcn.kernel @readfirstlane_tid arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr()
      : () -> !amdgcn.sgpr<[? + 2]>

    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    // Per-lane byte offset
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_a ins %c2, %tid
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    // readfirstlane(tid_x) -> should give lane 0's tid_x = 0
    %s_dest = amdgcn.alloca : !amdgcn.sgpr
    %s_val = amdgcn.vop1.lane #amdgcn.inst<v_readfirstlane_b32> %s_dest, %tid
      : (!amdgcn.sgpr, !amdgcn.vgpr) -> !amdgcn.sgpr

    // Broadcast to all lanes
    %v_dest = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v_dest, %s_val
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr

    // Store: out[tid_x] = readfirstlane(tid_x)
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
