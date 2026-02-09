// E2E tests for scf-pipeline with original iter_args.
// Four small kernels covering {with,without} IV x {scalar,vgpr} iter_args.
// VGPR iter_args exercise the bufferization pass.

!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr

// 2-stage, scalar iter_arg, no iv.
amdgcn.module @test_iter_args_scalar_no_iv target = <gfx942> isa = <cdna3> {
  kernel @test_iter_args_scalar_no_iv arguments <[
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

      // Stage 1: accumulate constant into iter_arg (no IV)
      %new_acc = arith.addi %acc, %c7_i32 {sched.stage = 1 : i32} : i32
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

// 2-stage, scalar iter_arg, with iv
amdgcn.module @test_iter_args_scalar_with_iv target = <gfx942> isa = <cdna3> {
  kernel @test_iter_args_scalar_with_iv arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init = arith.constant 0 : i32

    %result = scf.for %i = %c0 to %c8 step %c1
        iter_args(%acc = %init) -> (i32) {
      // Stage 0: cast IV to i32 (uses induction variable)
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32

      // Stage 1: accumulate i into iter_arg
      %new_acc = arith.addi %acc, %i_i32 {sched.stage = 1 : i32} : i32
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

// 2-stage, VGPR iter_arg (exercises bufferization), no iv
amdgcn.module @test_iter_args_vgpr_no_iv target = <gfx942> isa = <cdna3> {
  kernel @test_iter_args_vgpr_no_iv arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %init_i32 = arith.constant 0 : i32

    // Create initial vgpr value (0)
    %init_s = lsir.to_reg %init_i32 : i32 -> !amdgcn.sgpr
    %d_init = amdgcn.alloca : !v
    %init_v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_init, %init_s : (!v, !amdgcn.sgpr) -> !v

    // Create constant 5 in vgpr
    %c5_s = lsir.to_reg %c5_i32 : i32 -> !amdgcn.sgpr
    %d_five = amdgcn.alloca : !v
    %five_v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_five, %c5_s : (!v, !amdgcn.sgpr) -> !v

    %result = scf.for %i = %c0 to %c4 step %c1
        iter_args(%acc = %init_v) -> (!v) {
      // Stage 0: independent work (no IV, no iter_arg)
      %scratch = lsir.to_reg %c3_i32 {sched.stage = 0 : i32} : i32 -> !v

      // Stage 1: acc = acc + 5 in vgpr (no IV, uses iter_arg)
      %d_add = amdgcn.alloca : !v
      %new_acc = amdgcn.vop2 v_add_u32 outs %d_add ins %acc, %five_v {sched.stage = 1 : i32}
        : !v, !v, !v
      scf.yield %new_acc : !v
    }

    // Store result at output[0]
    %c0_i32 = arith.constant 0 : i32
    %off = lsir.to_reg %c0_i32 : i32 -> !v
    %tok = amdgcn.store global_store_dword data %result addr %out offset d(%off)
      : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// 2-stage, VGPR iter_arg (exercises bufferization), with iv
// Combines IV dependence, cross-stage value, VGPR iter_arg, and bufferization.
amdgcn.module @test_iter_args_vgpr_with_iv target = <gfx942> isa = <cdna3> {
  kernel @test_iter_args_vgpr_with_iv arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out = load_arg 0 : !sx2
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init_i32 = arith.constant 0 : i32

    // Create initial vgpr value (0)
    %init_s = lsir.to_reg %init_i32 : i32 -> !amdgcn.sgpr
    %d_init = amdgcn.alloca : !v
    %init_v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_init, %init_s : (!v, !amdgcn.sgpr) -> !v

    %result = scf.for %i = %c0 to %c8 step %c1
        iter_args(%acc = %init_v) -> (!v) {
      // Stage 0: cast IV to vgpr
      %i_i32 = arith.index_cast %i {sched.stage = 0 : i32} : index to i32
      %i_s = lsir.to_reg %i_i32 {sched.stage = 0 : i32} : i32 -> !amdgcn.sgpr
      %d_i = amdgcn.alloca : !v
      %i_v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d_i, %i_s {sched.stage = 0 : i32}
        : (!v, !amdgcn.sgpr) -> !v

      // Stage 1: acc = acc + i_v (cross-stage + iter_arg)
      %d_add = amdgcn.alloca : !v
      %new_acc = amdgcn.vop2 v_add_u32 outs %d_add ins %acc, %i_v {sched.stage = 1 : i32}
        : !v, !v, !v
      scf.yield %new_acc : !v
    }

    // Store result at output[0]
    %c0_i32 = arith.constant 0 : i32
    %off = lsir.to_reg %c0_i32 : i32 -> !v
    %tok = amdgcn.store global_store_dword data %result addr %out offset d(%off)
      : ins(!v, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    end_kernel
  }
}

// Six-stage pipeline (stages 0-5) over 12 iterations with accumulator.
// Stage 0: to_reg, stage 1: alloca+vop1 (unused vgpr, cross-stage lifetime),
// stages 2-4: independent scalar work (exercises prologue/epilogue depth),
// stage 5: accumulate constant 5 into scalar iter_arg.
amdgcn.module @test_scf_pipeline_iter_args target = <gfx942> isa = <cdna3> {
  kernel @test_iter_args arguments <[#amdgcn.buffer_arg<address_space = generic>]> {
    %out = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    wait lgkm_cnt 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c5_i32 = arith.constant 5 : i32
    %c7_i32 = arith.constant 7 : i32
    %init = arith.constant 0 : i32

    %result = scf.for %i = %c0 to %c12 step %c1
        iter_args(%acc = %init) -> (i32) {
      // Stage 0: prepare a constant in sgpr
      %s0 = lsir.to_reg %c7_i32 {sched.stage = 0 : i32} : i32 -> !amdgcn.sgpr

      // Stage 1: alloca + move to vgpr (unused, exercises cross-stage lifetime)
      %d0 = amdgcn.alloca : !amdgcn.vgpr
      %v0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %d0, %s0 {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr

      // Stage 2: independent scalar work
      %s2 = lsir.to_reg %c5_i32 {sched.stage = 2 : i32} : i32 -> !amdgcn.sgpr

      // Stage 3: independent scalar work
      %s3 = lsir.to_reg %c7_i32 {sched.stage = 3 : i32} : i32 -> !amdgcn.sgpr

      // Stage 4: independent scalar work
      %s4 = lsir.to_reg %c5_i32 {sched.stage = 4 : i32} : i32 -> !amdgcn.sgpr

      // Stage 5: accumulate constant 5 into scalar iter_arg
      %new_acc = arith.addi %acc, %c5_i32 {sched.stage = 5 : i32} : i32
      scf.yield %new_acc : i32
    }

    // Store final sum (60) -- all lanes write the same value
    // byte offset 60 = int32 index 15
    %s = lsir.to_reg %result : i32 -> !amdgcn.sgpr
    %d = amdgcn.alloca : !amdgcn.vgpr
    %v = amdgcn.vop1.vop1 <v_mov_b32_e32> %d, %s : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %tok = amdgcn.store global_store_dword data %v addr %out offset d(%v) : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.write_token<flat>
    end_kernel
  }
}
