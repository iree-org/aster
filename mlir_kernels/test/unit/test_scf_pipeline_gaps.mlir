!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr

// Two-stage pipeline with gap: stages {0, 2}.
amdgcn.module @test_gap_0_2 target = <gfx942> isa = <cdna3> {
  kernel @test_gap_0_2 arguments <[
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

      // Stage 2: compute offset = i * 4, store val
      %i_i32_s2 = arith.index_cast %i {sched.stage = 2 : i32} : index to i32
      %off = arith.muli %i_i32_s2, %c4_i32 {sched.stage = 2 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 2 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s_val, %val_reg {sched.stage = 2 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 2 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Two-stage pipeline with wide gap: stages {0, 3}.
amdgcn.module @test_gap_0_3 target = <gfx942> isa = <cdna3> {
  kernel @test_gap_0_3 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s_val = alloca : !v
    %s_off = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32

    scf.for %i = %c0 to %c8 step %c1 {
      // Stage 0: compute val = i * 5
      %i_i32_s0 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %val = arith.muli %i_i32_s0, %c5_i32 {sched.stage = 0 : i32} : i32
      %val_reg = lsir.to_reg %val {sched.stage = 0 : i32} : i32 -> !v

      // Stage 3: compute offset = i * 4, store val
      %i_i32_s3 = arith.index_cast %i {sched.stage = 3 : i32} : index to i32
      %off = arith.muli %i_i32_s3, %c4_i32 {sched.stage = 3 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 3 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s_val, %val_reg {sched.stage = 3 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 3 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Three-stage pipeline with gaps: stages {0, 2, 5}.
amdgcn.module @test_gap_0_2_5 target = <gfx942> isa = <cdna3> {
  kernel @test_gap_0_2_5 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %s0 = alloca : !v
    %s2 = alloca : !v
    %s_off = alloca : !v

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10_idx = arith.constant 10 : index
    %c4_i32 = arith.constant 4 : i32
    %c10_i32 = arith.constant 10 : i32

    scf.for %i = %c0 to %c10_idx step %c1 {
      // Stage 0: val = i
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %v0 = lsir.to_reg %i_i32 {sched.stage = 0 : i32} : i32 -> !v

      // Stage 2: val = val + 10
      %v0_i32 = lsir.from_reg %v0 {sched.stage = 2 : i32} : !v -> i32
      %sum = arith.addi %v0_i32, %c10_i32 {sched.stage = 2 : i32} : i32
      %v2 = lsir.to_reg %sum {sched.stage = 2 : i32} : i32 -> !v

      // Stage 5: store val to output[i]
      %i_i32_s5 = arith.index_cast %i {sched.stage = 5 : i32} : index to i32
      %off = arith.muli %i_i32_s5, %c4_i32 {sched.stage = 5 : i32} : i32
      %off_reg = lsir.to_reg %off {sched.stage = 5 : i32} : i32 -> !v
      %data = amdgcn.vop1.vop1 <v_mov_b32_e32> %s2, %v2 {sched.stage = 5 : i32}
        : (!v, !v) -> !v
      %tok = amdgcn.store global_store_dword data %data addr %out_ptr offset d(%off_reg) {sched.stage = 5 : i32}
        : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    }

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Two-stage pipeline with gap {0, 2} and scalar iter_arg accumulator.
amdgcn.module @test_gap_0_2_iter_args target = <gfx942> isa = <cdna3> {
  kernel @test_gap_0_2_iter_args arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %init = arith.constant 0 : i32

    %result = scf.for %i = %c0 to %c6 step %c1
        iter_args(%acc = %init) -> (i32) {
      // Stage 0: independent work (no IV, no iter_arg use)
      %s0 = lsir.to_reg %c3_i32 {sched.stage = 0 : i32} : i32 -> !v

      // Stage 2: accumulate constant into iter_arg (no IV)
      %new_acc = arith.addi %acc, %c7_i32 {sched.stage = 2 : i32} : i32
      scf.yield %new_acc : i32
    }

    // Store result at output[0]
    %s = lsir.to_reg %result : i32 -> !amdgcn.sgpr
    %d = amdgcn.alloca : !v
    %v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d, %s : (!v, !amdgcn.sgpr) -> !v
    %c0_i32 = arith.constant 0 : i32
    %off = lsir.to_reg %c0_i32 : i32 -> !v
    %tok = amdgcn.store global_store_dword data %v addr %out offset d(%off)
      : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}
