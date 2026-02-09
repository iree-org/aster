// End-to-end tests for scf-pipeline with LDS double-buffering.
// Each kernel writes data to LDS in one stage and reads it back in a later
// stage, with the pipeliner automatically introducing double-buffered LDS
// allocs via multibuffer prep.

!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr

// Two-stage LDS pass-through (no IV dependence in LDS path).
amdgcn.module @test_lds_passthrough target = <gfx942> isa = <cdna3> {
  kernel @test_lds_passthrough arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32

    // Move 42 into a vgpr for ds_write
    %d_val = amdgcn.alloca : !v
    %val = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_val, %c42_i32 : (!v, i32) -> !v

    %d_store = amdgcn.alloca : !v

    scf.for %i = %c0 to %c4 step %c1 {
      // Stage 0: alloc LDS, write 42 at offset 0 within the LDS buffer
      %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
      %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
      %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !v
      %wtok = amdgcn.store ds_write_b32 data %val addr %lds_addr offset c(%c0_i32)
        {sched.stage = 0 : i32}
        : ins(!v, !v, i32) -> !amdgcn.write_token<shared>

      // Stage 1: wait, read back, store to output[0]
      amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
      %r_dest = amdgcn.alloca {sched.stage = 1 : i32} : !v
      %from_lds, %rtok = amdgcn.load ds_read_b32 dest %r_dest addr %lds_addr
        {sched.stage = 1 : i32}
        : dps(!v) ins(!v) -> !amdgcn.read_token<shared>
      amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>

      // Store to output[0] -- use zero offset for all lanes
      %off_reg = lsir.to_reg %c0_i32 {sched.stage = 1 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_store, %from_lds
        {sched.stage = 1 : i32} : (!v, !v) -> !v
      %stok = amdgcn.store global_store_dword data %data addr %out_ptr
        offset d(%off_reg) {sched.stage = 1 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>

      amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Two-stage LDS with IV-dependent data.
amdgcn.module @test_lds_iv_dep target = <gfx942> isa = <cdna3> {
  kernel @test_lds_iv_dep arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c7_i32 = arith.constant 7 : i32

    %d_val = amdgcn.alloca : !v
    %d_store = amdgcn.alloca : !v

    scf.for %i = %c0 to %c8 step %c1 {
      // Stage 0: compute val = i * 7, write to LDS
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %val = arith.muli %i_i32, %c7_i32 {sched.stage = 0 : i32} : i32
      %val_reg = lsir.to_reg %val {sched.stage = 0 : i32} : i32 -> !v

      %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
      %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
      %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !v
      %wtok = amdgcn.store ds_write_b32 data %val_reg addr %lds_addr offset c(%c0_i32)
        {sched.stage = 0 : i32}
        : ins(!v, !v, i32) -> !amdgcn.write_token<shared>

      // Stage 1: wait, read from LDS, store to output[i]
      amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
      %r_dest = amdgcn.alloca {sched.stage = 1 : i32} : !v
      %from_lds, %rtok = amdgcn.load ds_read_b32 dest %r_dest addr %lds_addr
        {sched.stage = 1 : i32}
        : dps(!v) ins(!v) -> !amdgcn.read_token<shared>
      amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>

      // Store to output[i*4] (byte offset)
      %i_i32_s1 = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
      %off = arith.muli %i_i32_s1, %c4_i32 {sched.stage = 1 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 1 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_store, %from_lds
        {sched.stage = 1 : i32} : (!v, !v) -> !v
      %stok = amdgcn.store global_store_dword data %data addr %out_ptr
        offset d(%off_reg) {sched.stage = 1 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>

      amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Two-stage LDS with accumulator iter_arg.
amdgcn.module @test_lds_accum target = <gfx942> isa = <cdna3> {
  kernel @test_lds_accum arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32

    // Write constant 3 into a vgpr
    %d_val = amdgcn.alloca : !v
    %val = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_val, %c3_i32 : (!v, i32) -> !v

    // Accumulator init = 0
    %d_acc = amdgcn.alloca : !v
    %acc_init = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_acc, %c0_i32 : (!v, i32) -> !v

    %d_add = amdgcn.alloca : !v

    %final_acc = scf.for %i = %c0 to %c6 step %c1
        iter_args(%acc = %acc_init) -> (!v) {
      // Stage 0: write 3 to LDS
      %lds = amdgcn.alloc_lds 256 {sched.stage = 0 : i32}
      %lds_off = amdgcn.get_lds_offset %lds {sched.stage = 0 : i32} : i32
      %lds_addr = lsir.to_reg %lds_off {sched.stage = 0 : i32} : i32 -> !v
      %wtok = amdgcn.store ds_write_b32 data %val addr %lds_addr offset c(%c0_i32)
        {sched.stage = 0 : i32}
        : ins(!v, !v, i32) -> !amdgcn.write_token<shared>

      // Stage 1: read from LDS, accumulate
      amdgcn.wait deps %wtok {sched.stage = 1 : i32} : !amdgcn.write_token<shared>
      %r_dest = amdgcn.alloca {sched.stage = 1 : i32} : !v
      %from_lds, %rtok = amdgcn.load ds_read_b32 dest %r_dest addr %lds_addr
        {sched.stage = 1 : i32}
        : dps(!v) ins(!v) -> !amdgcn.read_token<shared>
      amdgcn.wait deps %rtok {sched.stage = 1 : i32} : !amdgcn.read_token<shared>

      %new_acc = amdgcn.vop2 v_add_u32 outs %d_add
        ins %acc, %from_lds {sched.stage = 1 : i32} : !v, !v, !v

      amdgcn.dealloc_lds %lds {sched.stage = 1 : i32}
      scf.yield %new_acc : !v
    }

    // Store final accumulator to output[0]
    %off_reg = lsir.to_reg %c0_i32 : i32 -> !v
    %stok = amdgcn.store global_store_dword data %final_acc addr %out_ptr
      offset d(%off_reg)
      : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}
