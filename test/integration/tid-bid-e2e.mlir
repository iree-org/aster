// RUN: aster-opt %s --verify-roundtrip
//
// E2E test: verify packed workitem ID extraction and block ID handling.
//
// On CDNA3/CDNA4, workitem IDs are packed in VGPR0 as {Z[9:0], Y[9:0], X[9:0]}.
// This test verifies that thread_id x/y/z and block_id x/y/z are correctly
// extracted and stored to global memory via linear indexing.
//
// Test values:
//   thread_id: x=42 (1D/2D), x=10,y=5,z=7 (3D, fits 1024-thread WG limit)
//   block_id:  x=42,y=5,z=7

amdgcn.module @tid_bid_mod target = #amdgcn.target<gfx942> {

  // Helper: load output pointer from kernel arg 0.
  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %out_ptr : !amdgcn.sgpr<[? + 2]>
  }

  // ----- Test 1: thread_id x only (no masking needed) -----
  // block=(64,1,1), grid=(1,1,1). Output: 64 dwords.
  // Each thread stores tid_x at output[tid_x].
  amdgcn.kernel @tid_x_only arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c2, %tid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %tok = amdgcn.store global_store_dword data %tid_x addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 2: thread_id x, y (packed extraction with mask) -----
  // block=(64,8,1), grid=(1,1,1). Output: 512*2 = 1024 dwords.
  // Each thread stores [tid_x, tid_y] at output[linear*2..linear*2+1]
  // where linear = (tid_y << 6) | tid_x.
  amdgcn.kernel @tid_xy arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_y = amdgcn.thread_id y : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32

    // linear = (tid_y << 6) | tid_x  (block_x=64=2^6, bits don't overlap)
    %c6 = arith.constant 6 : i32
    %sy_a = amdgcn.alloca : !amdgcn.vgpr
    %shifted_y = amdgcn.v_lshlrev_b32 outs(%sy_a) ins(%c6, %tid_y) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %lin_a = amdgcn.alloca : !amdgcn.vgpr
    %linear = amdgcn.v_or_b32 outs(%lin_a) ins(%shifted_y, %tid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)

    // byte offset = linear * 8 = linear << 3 (2 dwords per thread)
    %c3 = arith.constant 3 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c3, %linear) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %tok0 = amdgcn.store global_store_dword data %tid_x addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok1 = amdgcn.store global_store_dword data %tid_y addr %out_ptr
      offset d(%voffset) + c(%c4)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 3: thread_id x, y, z (full packed extraction) -----
  // block=(16,8,8), grid=(1,1,1). Output: 1024*3 = 3072 dwords.
  // Each thread stores [tid_x, tid_y, tid_z] at output[linear*3..linear*3+2]
  // where linear = (tid_z << 7) | (tid_y << 4) | tid_x.
  // (block_x=16=2^4, block_x*block_y=128=2^7, bits don't overlap)
  amdgcn.kernel @tid_xyz arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_y = amdgcn.thread_id y : !amdgcn.vgpr
    %tid_z = amdgcn.thread_id z : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32

    // linear = (tid_z << 7) | (tid_y << 4) | tid_x
    %c4_shift = arith.constant 4 : i32
    %c7 = arith.constant 7 : i32
    %sy_a = amdgcn.alloca : !amdgcn.vgpr
    %shifted_y = amdgcn.v_lshlrev_b32 outs(%sy_a) ins(%c4_shift, %tid_y) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %sz_a = amdgcn.alloca : !amdgcn.vgpr
    %shifted_z = amdgcn.v_lshlrev_b32 outs(%sz_a) ins(%c7, %tid_z) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %t1_a = amdgcn.alloca : !amdgcn.vgpr
    %t1 = amdgcn.v_or_b32 outs(%t1_a) ins(%shifted_y, %tid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %lin_a = amdgcn.alloca : !amdgcn.vgpr
    %linear = amdgcn.v_or_b32 outs(%lin_a) ins(%shifted_z, %t1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)

    // byte offset = linear * 12 (3 dwords per thread)
    %c12 = arith.constant 12 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_mul_lo_u32 outs(%voff_a) ins(%linear, %c12) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32)

    %tok0 = amdgcn.store global_store_dword data %tid_x addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok1 = amdgcn.store global_store_dword data %tid_y addr %out_ptr
      offset d(%voffset) + c(%c4)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok2 = amdgcn.store global_store_dword data %tid_z addr %out_ptr
      offset d(%voffset) + c(%c8)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 4: block_id x, y, z (SGPR system registers) -----
  // grid=(43,6,8), block=(64,1,1). Output: 2064*3 = 6192 dwords.
  // Each WG stores [bid_x, bid_y, bid_z] at output[linear_bid*3..+2]
  // where linear_bid = bid_x + 43 * (bid_y + 6 * bid_z).
  // All 64 lanes write identical data (block_id is uniform) -- benign race.
  amdgcn.kernel @bid_xyz arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %bid_x = amdgcn.block_id x : !amdgcn.sgpr
    %bid_y = amdgcn.block_id y : !amdgcn.sgpr
    %bid_z = amdgcn.block_id z : !amdgcn.sgpr

    // Broadcast block IDs to VGPRs for arithmetic and store.
    %vbx_a = amdgcn.alloca : !amdgcn.vgpr
    %v_bid_x = amdgcn.v_mov_b32 outs(%vbx_a) ins(%bid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %vby_a = amdgcn.alloca : !amdgcn.vgpr
    %v_bid_y = amdgcn.v_mov_b32 outs(%vby_a) ins(%bid_y) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)
    %vbz_a = amdgcn.alloca : !amdgcn.vgpr
    %v_bid_z = amdgcn.v_mov_b32 outs(%vbz_a) ins(%bid_z) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    // linear_bid = bid_x + 43 * (bid_y + 6 * bid_z)
    // Factored to avoid literal constants > 64 (VOP3 inline limit).
    %c6 = arith.constant 6 : i32
    %c43 = arith.constant 43 : i32
    %t1_a = amdgcn.alloca : !amdgcn.vgpr
    %t1 = amdgcn.v_mul_lo_u32 outs(%t1_a) ins(%v_bid_z, %c6) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32)
    %t2_a = amdgcn.alloca : !amdgcn.vgpr
    %t2 = amdgcn.v_add_u32 outs(%t2_a) ins(%t1, %v_bid_y) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)
    %t3_a = amdgcn.alloca : !amdgcn.vgpr
    %t3 = amdgcn.v_mul_lo_u32 outs(%t3_a) ins(%t2, %c43) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32)
    %lin_bid_a = amdgcn.alloca : !amdgcn.vgpr
    %linear_bid = amdgcn.v_add_u32 outs(%lin_bid_a) ins(%t3, %v_bid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)

    // byte offset = linear_bid * 12 (3 dwords per block)
    %c12 = arith.constant 12 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_mul_lo_u32 outs(%voff_a) ins(%linear_bid, %c12) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32)

    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32

    %tok0 = amdgcn.store global_store_dword data %v_bid_x addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok1 = amdgcn.store global_store_dword data %v_bid_y addr %out_ptr
      offset d(%voffset) + c(%c4)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok2 = amdgcn.store global_store_dword data %v_bid_z addr %out_ptr
      offset d(%voffset) + c(%c8)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // ----- Test 5: thread_id x + block_id x (combined) -----
  // grid=(43,1,1), block=(64,1,1). Output: 43*64*2 = 5504 dwords.
  // Each thread stores [tid_x, bid_x] at output[linear*2..linear*2+1]
  // where linear = (bid_x << 6) | tid_x.
  amdgcn.kernel @tid_x_bid_x arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_output_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %bid_x = amdgcn.block_id x : !amdgcn.sgpr

    %vbx_a = amdgcn.alloca : !amdgcn.vgpr
    %v_bid_x = amdgcn.v_mov_b32 outs(%vbx_a) ins(%bid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    // linear = (v_bid_x << 6) | tid_x  (block_x=64=2^6)
    %c6 = arith.constant 6 : i32
    %sbid_a = amdgcn.alloca : !amdgcn.vgpr
    %shifted_bid = amdgcn.v_lshlrev_b32 outs(%sbid_a) ins(%c6, %v_bid_x) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %lin_a = amdgcn.alloca : !amdgcn.vgpr
    %linear = amdgcn.v_or_b32 outs(%lin_a) ins(%shifted_bid, %tid_x) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr)

    // byte offset = linear * 8 = linear << 3 (2 dwords per thread)
    %c3 = arith.constant 3 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_a) ins(%c3, %linear) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32

    %tok0 = amdgcn.store global_store_dword data %tid_x addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    %tok1 = amdgcn.store global_store_dword data %v_bid_x addr %out_ptr
      offset d(%voffset) + c(%c4)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
