// RUN: aster-opt %s --verify-roundtrip
//
// E2E test: write constant 42 to M0 special register, read it back to SGPR,
// broadcast to VGPR, store to global buffer.
//
// Verifies the M0 -> SGPR -> VGPR -> global memory pipeline works end-to-end.
// Each lane stores the constant 42 to its slot in the output buffer.

amdgcn.module @m0_roundtrip_mod target = #amdgcn.target<gfx942> {

  func.func private @load_output_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %out_ptr : !amdgcn.sgpr<[? + 2]>
  }

  amdgcn.kernel @m0_roundtrip_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {

    %out_ptr = func.call @load_output_ptr()
      : () -> !amdgcn.sgpr<[? + 2]>

    %tid = amdgcn.thread_id x : !amdgcn.vgpr

    // Compute per-lane byte offset: threadidx.x * 4
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)

    // Write constant 42 to M0 via s_mov_b32
    // M0 is pre-allocated (fixed physical register), so write has no SSA result.
    %m0 = amdgcn.alloca : !amdgcn.m0<0>
    %c42 = arith.constant 42 : i32
    amdgcn.sop1 s_mov_b32 outs %m0 ins %c42
      : !amdgcn.m0<0>, i32

    // Read M0 back into an SGPR via s_mov_b32
    %s_dest = amdgcn.alloca : !amdgcn.sgpr
    %s_val = amdgcn.sop1 s_mov_b32 outs %s_dest ins %m0
      : !amdgcn.sgpr, !amdgcn.m0<0>

    // Broadcast scalar to all VGPR lanes via v_mov_b32_e32
    %v_dest = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.v_mov_b32 outs(%v_dest) ins(%s_val) : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr)

    // Store to global: out[threadidx.x] = 42
    %c0 = arith.constant 0 : i32
    %tok_st = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
