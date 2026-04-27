// Register allocation: out[tid] = in[tid] + 42, with VIRTUAL registers.
// Compare to 03_vector_add where every register is hand-allocated.
// Here we use !amdgcn.vgpr (no number) and let the compiler assign them.
//
// Key differences from hand-allocated examples:
//   alloca : !amdgcn.vgpr    -- virtual (compiler picks the number)
//   alloca : !amdgcn.vgpr<11> -- pre-allocated (pinned to v11, regalloc works around it)
//   load_arg N               -- virtual SGPR pair for kernel argument N
//   thread_id x              -- virtual VGPR for thread ID
//   wait deps %tok           -- token-based wait (compiler lowers to s_waitcnt)
//   vop2 returns a value     -- SSA data flow, not side-effecting register writes
//
// The amdgcn-backend pass runs register allocation (graph coloring),
// converts token waits to s_waitcnt, and legalizes control flow.
// Pre-allocated registers are treated as constraints: the allocator
// assigns virtual registers to physical slots that don't conflict.

amdgcn.module @regalloc target = <gfx942> {
  amdgcn.kernel @kernel arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.buffer_arg<address_space = generic, access = write_only>
    ]> {
    // Virtual SGPR pairs for buffer pointers
    %src = amdgcn.load_arg 0 : !amdgcn.sgpr<[? : ? + 2]>
    %dst = amdgcn.load_arg 1 : !amdgcn.sgpr<[? : ? + 2]>

    // Thread ID and byte offset = tid * 4
    // Pin the offset to v11 -- regalloc works around pre-allocated registers.
    %tid = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %off_reg = amdgcn.alloca : !amdgcn.vgpr<11>
    amdgcn.vop2 v_lshlrev_b32_e32 outs %off_reg ins %c2, %tid
      : !amdgcn.vgpr<11>, i32, !amdgcn.vgpr

    // Load in[tid]
    %data_reg = amdgcn.alloca : !amdgcn.vgpr
    %c0 = arith.constant 0 : i32
    %data, %tok_ld = amdgcn.load global_load_dword dest %data_reg addr %src
      offset d(%off_reg) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<11>, i32)
        -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok_ld : !amdgcn.read_token<flat>

    // out[tid] = in[tid] + 42
    %c42 = arith.constant 42 : i32
    %result_reg = amdgcn.alloca : !amdgcn.vgpr
    %result = amdgcn.vop2 v_add_u32 outs %result_reg ins %c42, %data
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    // Store result
    %tok_st = amdgcn.store global_store_dword data %result addr %dst
      offset d(%off_reg) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vgpr<11>, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.wait deps %tok_st : !amdgcn.write_token<flat>

    amdgcn.end_kernel
  }
}
