// RUN: aster-opt %s --verify-roundtrip
//
// E2E test for the VOPC comparison path in LegalizeCF.
//
// Uses lsir.cmpi + lsir.select with VGPR operands to exercise the VOPC
// lowering path (v_cmp_* + v_cndmask_b32). The select is per-lane:
// each lane independently picks true_val or false_val based on its
// VCC bit from the compare.
//
// Three scenarios:
//   always_true:  tid < 100 (all 64 lanes true)  -> all lanes output 42.
//   always_false: tid < 0   (no lane true)        -> all lanes output 99.
//   per_lane:     tid < 32  (lanes 0-31 true)     -> lanes 0-31 output 42,
//                                                    lanes 32-63 output 99.

amdgcn.module @vopc_select_mod target = #amdgcn.target<gfx942> {

  func.func private @load_out_ptr() -> !amdgcn.sgpr<[? + 2]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    amdgcn.s_waitcnt lgkmcnt = 0
    return %out_ptr : !amdgcn.sgpr<[? + 2]>
  }

  // All lanes: tid < 100 -> true -> select 42.
  amdgcn.kernel @vopc_select_always_true_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_out_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %c100 = arith.constant 100 : i32
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    // Materialize constants into VGPRs (v_cndmask_b32 needs VGPR operands).
    %v_true_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_true = amdgcn.v_mov_b32 outs(%v_true_alloc) ins(%c42) : outs(!amdgcn.vgpr) ins(i32)
    %v_false_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_false = amdgcn.v_mov_b32 outs(%v_false_alloc) ins(%c99) : outs(!amdgcn.vgpr) ins(i32)
    %v_out_alloc = amdgcn.alloca : !amdgcn.vgpr
    %cmp = lsir.cmpi i32 slt %tid, %c100 : !amdgcn.vgpr, i32
    %selected = lsir.select %v_out_alloc, %cmp, %v_true, %v_false
      : !amdgcn.vgpr, i1, !amdgcn.vgpr, !amdgcn.vgpr
    %tok = amdgcn.global_store_dword data %selected addr %out_ptr offset d(%voffset) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // No lane: tid < 0 -> false -> select 99.
  amdgcn.kernel @vopc_select_always_false_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_out_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %v_true_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_true = amdgcn.v_mov_b32 outs(%v_true_alloc) ins(%c42) : outs(!amdgcn.vgpr) ins(i32)
    %v_false_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_false = amdgcn.v_mov_b32 outs(%v_false_alloc) ins(%c99) : outs(!amdgcn.vgpr) ins(i32)
    %v_out_alloc = amdgcn.alloca : !amdgcn.vgpr
    %cmp = lsir.cmpi i32 slt %tid, %c0 : !amdgcn.vgpr, i32
    %selected = lsir.select %v_out_alloc, %cmp, %v_true, %v_false
      : !amdgcn.vgpr, i1, !amdgcn.vgpr, !amdgcn.vgpr
    %tok = amdgcn.global_store_dword data %selected addr %out_ptr offset d(%voffset) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }

  // Per-lane: tid < 32 -> lanes 0-31 get 42, lanes 32-63 get 99.
  // Tests per-lane VCC-based select (v_cndmask_b32).
  amdgcn.kernel @vopc_select_per_lane_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %out_ptr = func.call @load_out_ptr() : () -> !amdgcn.sgpr<[? + 2]>
    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %c32 = arith.constant 32 : i32
    %voff_alloc = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.v_lshlrev_b32 outs(%voff_alloc) ins(%c2, %tid) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr)
    %v_true_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_true = amdgcn.v_mov_b32 outs(%v_true_alloc) ins(%c42) : outs(!amdgcn.vgpr) ins(i32)
    %v_false_alloc = amdgcn.alloca : !amdgcn.vgpr
    %v_false = amdgcn.v_mov_b32 outs(%v_false_alloc) ins(%c99) : outs(!amdgcn.vgpr) ins(i32)
    %v_out_alloc = amdgcn.alloca : !amdgcn.vgpr
    %cmp = lsir.cmpi i32 slt %tid, %c32 : !amdgcn.vgpr, i32
    %selected = lsir.select %v_out_alloc, %cmp, %v_true, %v_false
      : !amdgcn.vgpr, i1, !amdgcn.vgpr, !amdgcn.vgpr
    %tok = amdgcn.global_store_dword data %selected addr %out_ptr offset d(%voffset) + c(%c0) : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    amdgcn.s_waitcnt vmcnt = 0
    amdgcn.end_kernel
  }
}
