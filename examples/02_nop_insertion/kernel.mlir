// Hardware hazard: store reads v0 (thread ID), then we overwrite it.
// The memory unit is still reading v0 when the next instruction writes it.
// The amdgcn-hazards pass inserts v_nop delays to keep the pipeline safe.
//
// Each thread stores its thread ID to output[tid], then overwrites v0.

module {
  amdgcn.module @nop_demo target = <gfx942> {
    kernel @kernel arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = write_only>
    ]> attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    ^entry:
      // Hardware-initialized: s[0:1] = kernarg base, v0 = thread ID
      %s0 = alloca : !amdgcn.sgpr<0>
      %s1 = alloca : !amdgcn.sgpr<1>
      %v0 = alloca : !amdgcn.vgpr<0>
      %v1 = alloca : !amdgcn.vgpr<1>
      %s2 = alloca : !amdgcn.sgpr<2>
      %s3 = alloca : !amdgcn.sgpr<3>

      // Load output pointer from kernarg (implicitly "known" to be in `s[0:1]`)
      %kernarg = amdgcn.make_register_range %s0, %s1
        : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
      %out_ptr = amdgcn.make_register_range %s2, %s3
        : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
      %c0_i32_mig1 = arith.constant 0 : i32
      amdgcn.s_load_dwordx2 dest %out_ptr addr %kernarg offset c(%c0_i32_mig1) : outs(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>) mods(i32) -> !amdgcn.read_token<constant>
      amdgcn.s_waitcnt lgkmcnt = 0

      // Byte offset = tid * 4 (4 bytes per i32)
      %c2 = arith.constant 2 : i32
      amdgcn.v_lshlrev_b32 outs(%v1) ins(%c2, %v0) : outs(!amdgcn.vgpr<1>) ins(i32, !amdgcn.vgpr<0>)

      // Store thread ID (v0) to output[tid]
      %data_r = amdgcn.make_register_range %v0 : !amdgcn.vgpr<0>
      %c0 = arith.constant 0 : i32
      amdgcn.global_store_dword data %data_r addr %out_ptr offset d(%v1) + c(%c0) : ins(!amdgcn.vgpr<[0 : 1]>, !amdgcn.sgpr<[2 : 4]>, !amdgcn.vgpr<1>) mods(i32) -> !amdgcn.write_token<flat>

      // Immediately overwrite v0 -- HAZARD: memory still reading v0
      %c7 = arith.constant 7 : i32
      amdgcn.v_mov_b32 outs(%v0) ins(%c7) : outs(!amdgcn.vgpr<0>) ins(i32)

      // Self-check: v0 should be 7 after the overwrite
      %vcc = amdgcn.alloca : !amdgcn.vcc
      amdgcn.v_cmp_ne_i32 outs(%vcc) ins(%c7, %v0) : outs(!amdgcn.vcc) ins(i32, !amdgcn.vgpr<0>)
      amdgcn.cbranch s_cbranch_vccnz %vcc ^trap fallthrough(^ok)
        : !amdgcn.vcc

    ^ok:
      // Flush all vector memory operations
      amdgcn.s_waitcnt vmcnt = 0
      end_kernel

    ^trap:
      s_trap 2
      end_kernel
    }
  }
}
