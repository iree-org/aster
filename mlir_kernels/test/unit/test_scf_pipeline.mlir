!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr

// ============================================================================
// Two-stage pipeline, no IV dependence.
// Stage 0: move constant 42 to register (no IV use)
// Stage 1: store to output[0] (no IV use)
// Expected: output[0] = 42
// ============================================================================
amdgcn.module @test_two_stage_no_iv target = <gfx942> isa = <cdna3> {
  kernel @test_two_stage_no_iv arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s_store = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c42_i32 = arith.constant 42 : i32
    %c0_i32 = arith.constant 0 : i32

    scf.for %i = %c0 to %c8 step %c1 {
      // Stage 0: move constant to register (no IV use)
      %val_reg = lsir.to_reg %c42_i32 {sched.stage = 0 : i32} : i32 -> !v

      // Stage 1: store constant to output[0] (no IV use)
      %off_reg = lsir.to_reg %c0_i32 {sched.stage = 1 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s_store, %val_reg {sched.stage = 1 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 1 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// ============================================================================
// Two-stage pipeline, IV used only in stage 0.
// Stage 0: compute val = i * 4 (uses IV)
// Stage 1: store val to output at byte offset val (no IV use)
// Expected: output viewed as int32[8] = [0, 4, 8, 12, 16, 20, 24, 28]
// ============================================================================
amdgcn.module @test_two_stage_iv_s0_only target = <gfx942> isa = <cdna3> {
  kernel @test_two_stage_iv_s0_only arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s_store = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c4_i32 = arith.constant 4 : i32

    scf.for %i = %c0 to %c8 step %c1 {
      // Stage 0: compute val = i * 4
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %val = arith.muli %i_i32, %c4_i32 {sched.stage = 0 : i32} : i32
      %val_reg = lsir.to_reg %val {sched.stage = 0 : i32} : i32 -> !v

      // Stage 1: store val to output[val] (val is both the data and offset)
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s_store, %val_reg {sched.stage = 1 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%data) {sched.stage = 1 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// ============================================================================
// Two-stage pipeline with IV dependence in both stages.
// Stage 0: compute val = i * 3 (uses IV)
// Stage 1: compute offset = i * 4, store val to output[i] (uses IV)
// Tests that the kernel adjusts the IV correctly for stage 1.
// Expected: output[i] = i * 3, i.e. [0, 3, 6, 9, 12, 15, 18, 21]
// ============================================================================
amdgcn.module @test_two_stage_iv_dep target = <gfx942> isa = <cdna3> {
  kernel @test_two_stage_iv_dep arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s_val = alloca : !v
    %s_off = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32

    scf.for %i = %c0 to %c8 step %c1 {
      // Stage 0: compute val = i * 3
      %i_i32_s0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %val = arith.muli %i_i32_s0, %c3_i32 {sched.stage = 0 : i32} : i32
      %val_reg = lsir.to_reg %val {sched.stage = 0 : i32} : i32 -> !v

      // Stage 1: compute offset = i * 4, store val
      %i_i32_s1 = arith.index_cast %i {sched.stage = 1 : i32} : index to i32
      %off = arith.muli %i_i32_s1, %c4_i32 {sched.stage = 1 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 1 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s_val, %val_reg {sched.stage = 1 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 1 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// ============================================================================
// Five-stage pipeline (stages 0-4), linear value chain.
// Stage 0: val = i
// Stage 1: val = val + 1
// Stage 2: val = val + 10
// Stage 3: val = val + 100
// Stage 4: store val to output[i]
// Expected: output[i] = i + 111
//   [111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
//
// Stage 4 uses IV for the store offset, testing IV adjustment at depth 4.
// ============================================================================
amdgcn.module @test_five_stage target = <gfx942> isa = <cdna3> {
  kernel @test_five_stage arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s0 = alloca : !v
    %s1 = alloca : !v
    %s2 = alloca : !v
    %s3 = alloca : !v
    %s_off = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10_idx = arith.constant 10 : index
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant 10 : i32
    %c100_i32 = arith.constant 100 : i32

    scf.for %i = %c0 to %c10_idx step %c1 {
      // Stage 0: val = i
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %v0 = lsir.to_reg %i_i32 {sched.stage = 0 : i32} : i32 -> !v

      // Stage 1: val = val + 1
      %v0_i32 = lsir.from_reg %v0 {sched.stage = 1 : i32} : !v -> i32
      %sum1 = arith.addi %v0_i32, %c1_i32 {sched.stage = 1 : i32} : i32
      %v1 = lsir.to_reg %sum1 {sched.stage = 1 : i32} : i32 -> !v

      // Stage 2: val = val + 10
      %v1_i32 = lsir.from_reg %v1 {sched.stage = 2 : i32} : !v -> i32
      %sum2 = arith.addi %v1_i32, %c10_i32 {sched.stage = 2 : i32} : i32
      %v2 = lsir.to_reg %sum2 {sched.stage = 2 : i32} : i32 -> !v

      // Stage 3: val = val + 100
      %v2_i32 = lsir.from_reg %v2 {sched.stage = 3 : i32} : !v -> i32
      %sum3 = arith.addi %v2_i32, %c100_i32 {sched.stage = 3 : i32} : i32
      %v3 = lsir.to_reg %sum3 {sched.stage = 3 : i32} : i32 -> !v

      // Stage 4: store val to output[i]
      %i_i32_s4 = arith.index_cast %i {sched.stage = 4 : i32} : index to i32
      %off = arith.muli %i_i32_s4, %c4_i32 {sched.stage = 4 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 4 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s3, %v3 {sched.stage = 4 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 4 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}
