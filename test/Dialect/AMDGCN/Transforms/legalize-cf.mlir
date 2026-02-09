// RUN: aster-opt %s --amdgcn-legalize-cf --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: kernel @test_cond_branch_slt
// CHECK:         %[[A:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[B:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         cmpi s_cmp_lt_i32 outs %[[SCC]] ins %[[A]], %[[B]] : outs(!amdgcn.scc) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// Use SCC0 because ^bb1 (trueDest) is the next physical block - branch to ^bb2 if false
// CHECK:         cbranch s_cbranch_scc0 %[[SCC]] ^bb2 fallthrough(^bb1) : !amdgcn.scc
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_slt target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_cond_branch_slt {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %alloc0 = alloca : !amdgcn.sgpr<0>
    %alloc1 = alloca : !amdgcn.sgpr<1>
    %a = sop1 s_mov_b32 outs %alloc0 ins %c0_i32 : !amdgcn.sgpr<0>, i32
    %b = sop1 s_mov_b32 outs %alloc1 ins %c10_i32 : !amdgcn.sgpr<1>, i32
    %cmp = lsir.cmpi i32 slt %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    cf.cond_br %cmp, ^bb1, ^bb2
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @test_unconditional_branch
// CHECK:         branch s_branch ^bb1
// CHECK:       ^bb1:
// CHECK:         end_kernel
amdgcn.module @test_br target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_unconditional_branch {
    cf.br ^bb1
  ^bb1:
    end_kernel
  }
}

// -----

// Verify lsir.cmpi is converted to amdgcn.cmpi with allocated operands
// Verify cf.cond_br is converted to amdgcn.cbranch
// Entry check: use s_cbranch_scc0 because ^bb1 (trueDest) is the next physical block
// Branch to ^bb2 if SCC=0 (condition false), fallthrough to ^bb1 if SCC=1 (true)
// Verify block argument is removed (^bb1 has no args after legalization)
// Verify loop backedge uses same alloca
// Backedge: use s_cbranch_scc1 because ^bb2 (falseDest) is the next physical block
// Branch to ^bb1 if SCC=1 (continue loop), fallthrough to ^bb2 if SCC=0 (exit)

// CHECK-LABEL: kernel @test_cf_cond_br_lsir_cmpi
//       CHECK:   cmpi s_cmp_gt_i32 outs %{{.*}} ins %{{.*}}, %{{.*}} : outs(!amdgcn.scc) ins(!amdgcn.sgpr<6>, i32)
//       CHECK:   cbranch s_cbranch_scc0 %{{.*}} ^bb2 fallthrough(^bb1)
//       CHECK:   ^bb1:
//       CHECK:     sop2 s_add_u32 outs %[[LOOP_ALLOC:.*]] ins %[[LOOP_ALLOC]]
//       CHECK:     cmpi s_cmp_lt_i32 outs %{{.*}} ins %{{.*}}, %{{.*}} : outs(!amdgcn.scc) ins(!amdgcn.sgpr<7>, !amdgcn.sgpr<6>)
//       CHECK:     cbranch s_cbranch_scc1 %{{.*}} ^bb1 fallthrough(^bb2)
//       CHECK:   ^bb2:
//       CHECK:     end_kernel
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {

  amdgcn.kernel @test_cf_cond_br_lsir_cmpi arguments <[
        #amdgcn.buffer_arg<address_space = generic, access = read_only>,
        #amdgcn.buffer_arg<address_space = generic>
      ]>
      attributes {enable_workgroup_id_x = false}
  {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = alloca : !amdgcn.sgpr<0>
    %1 = alloca : !amdgcn.sgpr<1>
    %2 = alloca : !amdgcn.sgpr<2>
    %3 = alloca : !amdgcn.sgpr<3>
    %4 = alloca : !amdgcn.sgpr<4>
    %5 = alloca : !amdgcn.sgpr<5>
    %6 = alloca : !amdgcn.sgpr<6>
    %7 = alloca : !amdgcn.sgpr<7>
    %8 = alloca : !amdgcn.sgpr<8>
    %9 = alloca : !amdgcn.vgpr<0>

    %11 = make_register_range %0, %1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %12 = make_register_range %2, %3 : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %result, %token = load s_load_dwordx2 dest %12 addr %11 offset c(%c0_i32) : dps(!amdgcn.sgpr_range<[2 : 4]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
    %13 = make_register_range %4, %5 : !amdgcn.sgpr<4>, !amdgcn.sgpr<5>
    %result_0, %token_1 = load s_load_dwordx2 dest %13 addr %11 offset c(%c8_i32) : dps(!amdgcn.sgpr_range<[4 : 6]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0
    %result_2, %token_3 = load s_load_dword dest %6 addr %result : dps(!amdgcn.sgpr<6>) ins(!amdgcn.sgpr_range<[2 : 4]>) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0
    //
    // Loop start cond:
    %15 = lsir.cmpi i32 sgt %result_2, %c0_i32 : !amdgcn.sgpr<6>, i32
    // Loop iv: sgpr<7>
    %16 = sop1 s_mov_b32 outs %7 ins %c0_i32 : !amdgcn.sgpr<7>, i32
    cf.cond_br %15, ^bb1(%16 : !amdgcn.sgpr<7>), ^bb2
  ^bb1(%18: !amdgcn.sgpr<7>):  // 2 preds: ^bb0, ^bb1
    %19 = sop2 s_lshl_b32 outs %8 ins %18, %c2_i32 : !amdgcn.sgpr<8>, !amdgcn.sgpr<7>, i32
    %20 = amdgcn.vop1.vop1 <v_mov_b32_e32> %9, %19 : (!amdgcn.vgpr<0>, !amdgcn.sgpr<8>) -> !amdgcn.vgpr<0>
    %21 = store global_store_dword data %20 addr %result_0 offset d(%20) : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr_range<[4 : 6]>, !amdgcn.vgpr<0>) -> !amdgcn.write_token<flat>
    //
    // Loop iv increment: sgpr<7>
    %22 = sop2 s_add_u32 outs %7 ins %18, %c1_i32 : !amdgcn.sgpr<7>, !amdgcn.sgpr<7>, i32
    // Loop end cond: lsir.cmpi
    %24 = lsir.cmpi i32 slt %22, %result_2 : !amdgcn.sgpr<7>, !amdgcn.sgpr<6>
    // Loop backedge: cf.cond_br
    cf.cond_br %24, ^bb1(%22 : !amdgcn.sgpr<7>), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}

// -----

// Simple unconditional branch with 4-element VGPR range
// Verifies basic range decomposition and reconstruction at block entry
// The range should NOT appear as a block argument after legalization
// Instead, values should flow through allocas and be reconstructed at block entry
// CHECK-LABEL: kernel @test_br_vgpr_range_simple
amdgcn.module @test_br_vgpr_range target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_br_vgpr_range_simple {
    // Allocate constituent registers - verify they appear in order
    // CHECK:       %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
    // CHECK:       %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
    // CHECK:       %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
    // CHECK:       %[[V3:.*]] = alloca : !amdgcn.vgpr<3>
    %v0 = alloca : !amdgcn.vgpr<0>
    %v1 = alloca : !amdgcn.vgpr<1>
    %v2 = alloca : !amdgcn.vgpr<2>
    %v3 = alloca : !amdgcn.vgpr<3>

    // Initialize registers
    %c0 = arith.constant 0 : i32
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V0]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V1]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V2]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V3]]
    %init0 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v0, %c0 : (!amdgcn.vgpr<0>, i32) -> !amdgcn.vgpr<0>
    %init1 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v1, %c0 : (!amdgcn.vgpr<1>, i32) -> !amdgcn.vgpr<1>
    %init2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v2, %c0 : (!amdgcn.vgpr<2>, i32) -> !amdgcn.vgpr<2>
    %init3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v3, %c0 : (!amdgcn.vgpr<3>, i32) -> !amdgcn.vgpr<3>

    // Create range - uses vop1 results (which write to allocas)
    // CHECK:       make_register_range
    %range = make_register_range %init0, %init1, %init2, %init3 :
      !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>

    // Branch with range as operand
    // CHECK:       branch s_branch ^bb1
    cf.br ^bb1(%range : !amdgcn.vgpr_range<[0 : 4]>)

    // Block argument should be removed, range reconstructed
    // CHECK:       ^bb1:
    // CHECK-NOT:     ^bb1(%
    // Verify no duplicate allocas created
    // CHECK-NOT:     alloca
  ^bb1(%arg: !amdgcn.vgpr_range<[0 : 4]>):
    // Range should be reconstructed from SAME allocas at block entry
    // CHECK:       %[[RECONSTRUCTED:.*]] = make_register_range %[[V0]], %[[V1]], %[[V2]], %[[V3]]

    // Split the range - verify 4 results
    // CHECK:       %{{.*}}:4 = split_register_range %[[RECONSTRUCTED]]
    %split:4 = split_register_range %arg : !amdgcn.vgpr_range<[0 : 4]>

    // CHECK:       end_kernel
    end_kernel
  }
}

// -----

// Loop with VGPR range loop-carried value (MFMA accumulation pattern)
// This is the real-world use case: 4-register accumulator passed through iterations
// Verifies range decomposition in backedge and reconstruction at loop header
// CHECK-LABEL: kernel @test_loop_vgpr_range_carried
amdgcn.module @test_loop target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_loop_vgpr_range_carried {
    // Constants
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    // Allocate 4-register accumulator (v4-v7) - verify order
    // CHECK:       %[[V4:.*]] = alloca : !amdgcn.vgpr<4>
    // CHECK:       %[[V5:.*]] = alloca : !amdgcn.vgpr<5>
    // CHECK:       %[[V6:.*]] = alloca : !amdgcn.vgpr<6>
    // CHECK:       %[[V7:.*]] = alloca : !amdgcn.vgpr<7>
    %v4 = alloca : !amdgcn.vgpr<4>
    %v5 = alloca : !amdgcn.vgpr<5>
    %v6 = alloca : !amdgcn.vgpr<6>
    %v7 = alloca : !amdgcn.vgpr<7>

    // Loop counter
    // CHECK:       %[[S8:.*]] = alloca : !amdgcn.sgpr<8>
    %s8 = alloca : !amdgcn.sgpr<8>

    // Initialize accumulator to zero
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V4]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V5]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V6]]
    // CHECK:       vop1.vop1 <v_mov_b32_e32> %[[V7]]
    %init_v4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v4, %c0_i32 : (!amdgcn.vgpr<4>, i32) -> !amdgcn.vgpr<4>
    %init_v5 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v5, %c0_i32 : (!amdgcn.vgpr<5>, i32) -> !amdgcn.vgpr<5>
    %init_v6 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v6, %c0_i32 : (!amdgcn.vgpr<6>, i32) -> !amdgcn.vgpr<6>
    %init_v7 = amdgcn.vop1.vop1 <v_mov_b32_e32> %v7, %c0_i32 : (!amdgcn.vgpr<7>, i32) -> !amdgcn.vgpr<7>

    // Create initial accumulator range
    // CHECK:       make_register_range
    %acc_init = make_register_range %init_v4, %init_v5, %init_v6, %init_v7 :
      !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>

    // Initialize counter
    // CHECK:       sop1 s_mov_b32 outs %[[S8]]
    %counter_init = sop1 s_mov_b32 outs %s8 ins %c0_i32 : !amdgcn.sgpr<8>, i32

    // Branch to loop - passes counter (SGPR) and accumulator (VGPR range)
    // CHECK:       branch s_branch ^bb1
    cf.br ^bb1(%counter_init, %acc_init : !amdgcn.sgpr<8>, !amdgcn.vgpr_range<[4 : 8]>)

    // Loop header - block arguments should be removed
    // CHECK:       ^bb1:
    // CHECK-NOT:     ^bb1(%
    // Verify no duplicate allocas - counter flows through %[[S8]], accumulator through %[[V4]]-[[V7]]
    // CHECK-NOT:     alloca
  ^bb1(%counter: !amdgcn.sgpr<8>, %acc: !amdgcn.vgpr_range<[4 : 8]>):
    // Accumulator range should be reconstructed from SAME allocas at loop entry
    // CHECK:       %[[ACC_RECON:.*]] = make_register_range %[[V4]], %[[V5]], %[[V6]], %[[V7]]

    // Dummy operands for MFMA (simplified - real code would have loads)
    %v16 = alloca : !amdgcn.vgpr<16>
    %v17 = alloca : !amdgcn.vgpr<17>
    %v18 = alloca : !amdgcn.vgpr<18>
    %v19 = alloca : !amdgcn.vgpr<19>
    %dummy_a = make_register_range %v16, %v17 : !amdgcn.vgpr<16>, !amdgcn.vgpr<17>
    %dummy_b = make_register_range %v18, %v19 : !amdgcn.vgpr<18>, !amdgcn.vgpr<19>

    // MFMA: new_acc = MFMA(a, b, acc) - accumulator is both input and output
    // CHECK:       vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %[[ACC_RECON]]
    %new_acc = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16> %acc, %dummy_a, %dummy_b, %acc : <[16 : 18]>, <[18 : 20]>, !amdgcn.vgpr_range<[4 : 8]> -> !amdgcn.vgpr_range<[4 : 8]>

    // Increment counter - writes to %[[S8]] alloca
    // CHECK:       sop2 s_add_u32 outs %[[S8]] ins %[[S8]]
    %counter_next = sop2 s_add_u32 outs %s8 ins %counter, %c1_i32 : !amdgcn.sgpr<8>, !amdgcn.sgpr<8>, i32

    // Loop condition
    // CHECK:       cmpi s_cmp_lt_i32
    %cond = lsir.cmpi i32 slt %counter_next, %c2_i32 : !amdgcn.sgpr<8>, i32

    // Loop backedge - passes updated counter and accumulator to both loop and exit
    // CHECK:       cbranch s_cbranch_scc1 {{.*}} ^bb1 fallthrough(^bb2)
    cf.cond_br %cond, ^bb1(%counter_next, %new_acc : !amdgcn.sgpr<8>, !amdgcn.vgpr_range<[4 : 8]>), ^bb2(%new_acc : !amdgcn.vgpr_range<[4 : 8]>)

    // Exit block - receives final accumulator from loop
    // CHECK:       ^bb2:
    // CHECK-NOT:     ^bb2(%
    // CHECK-NOT:     alloca
  ^bb2(%final_acc: !amdgcn.vgpr_range<[4 : 8]>):
    // Reconstruct range at exit from SAME allocas
    // CHECK:       %[[FINAL_RECON:.*]] = make_register_range %[[V4]], %[[V5]], %[[V6]], %[[V7]]
    // Extract final values - verify 4 results
    // CHECK:       %{{.*}}:4 = split_register_range %[[FINAL_RECON]]
    %final:4 = split_register_range %final_acc : !amdgcn.vgpr_range<[4 : 8]>

    // CHECK:       end_kernel
    end_kernel
  }
}

// -----

// lsir.cmpi + lsir.select(i1) -> s_cmp + s_cselect_b32

// CHECK-LABEL: kernel @test_select_i1
// CHECK:         %[[A:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[B:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         cmpi s_cmp_eq_i32 outs %[[SCC]] ins %[[A]], %[[B]]
// CHECK:         sop2 s_cselect_b32 outs %{{.*}} ins
// CHECK:         end_kernel
amdgcn.module @test_select_i1_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_select_i1 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    %cmp = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %result = lsir.select %alloc2, %cmp, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    amdgcn.end_kernel
  }
}

// -----

// Test: multiple lsir.select from one lsir.cmpi (SCC fan-out)

// After dedup: exactly 1 alloca:scc, 1 cmpi, 2 s_cselect (sharing one SCC)
// CHECK-LABEL: kernel @test_select_fanout
// CHECK:         %[[A:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[B:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         cmpi s_cmp_eq_i32 outs %[[SCC]] ins %[[A]], %[[B]]
// CHECK-NOT:     alloca : !amdgcn.scc
// CHECK-NOT:     cmpi
// CHECK:         sop2 s_cselect_b32
// CHECK:         sop2 s_cselect_b32
// CHECK:         end_kernel
amdgcn.module @test_select_fanout_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_select_fanout {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %alloc3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    %cmp = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r1 = lsir.select %alloc2, %cmp, %c1, %c2 : !amdgcn.sgpr<2>, i1, i32, i32
    %r2 = lsir.select %alloc3, %cmp, %c3, %c4 : !amdgcn.sgpr<3>, i1, i32, i32
    amdgcn.end_kernel
  }
}

// -----

// Mixed consumers: exactly 1 alloca:scc, 1 cmpi, 1 s_cselect, 1 cbranch.

// CHECK-LABEL: kernel @test_mixed_consumers
// CHECK:         %[[A:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[B:.*]] = sop1 s_mov_b32 outs %{{.*}} ins
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         cmpi s_cmp_eq_i32 outs %[[SCC]] ins %[[A]], %[[B]]
// CHECK-NOT:     alloca : !amdgcn.scc
// CHECK-NOT:     cmpi
// CHECK:         sop2 s_cselect_b32
// CHECK:         cbranch s_cbranch_scc0 %[[SCC]]
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_mixed_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_mixed_consumers {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    %cmp = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r = lsir.select %alloc2, %cmp, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    cf.cond_br %cmp, ^bb1, ^bb2
  ^bb1:
    amdgcn.end_kernel
  ^bb2:
    amdgcn.end_kernel
  }
}

// -----

// Sequential non-overlapping i1 lifetimes are fine.

// CHECK-LABEL: kernel @test_sequential_i1
// CHECK:         cmpi s_cmp_eq_i32
// CHECK-NOT:     cmpi
// CHECK:         sop2 s_cselect_b32
// CHECK:         cmpi s_cmp_lt_i32
// CHECK-NOT:     cmpi
// CHECK:         sop2 s_cselect_b32
// CHECK:         end_kernel
amdgcn.module @test_sequential_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_sequential_i1 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %alloc3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    // First compare: consumed immediately by select
    %cmp1 = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r1 = lsir.select %alloc2, %cmp1, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    // Second compare: starts after first is consumed -- no overlap
    %cmp2 = lsir.cmpi i32 slt %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r2 = lsir.select %alloc3, %cmp2, %c42, %c99 : !amdgcn.sgpr<3>, i1, i32, i32
    amdgcn.end_kernel
  }
}

// -----

// Dead cmpi won't be lowered and doesn't consume SCC at runtime (would have
// been DCE'd away)

// CHECK-LABEL: kernel @test_dead_cmpi_then_live
// CHECK-NOT:     cmpi s_cmp_eq_i32
// CHECK:         cmpi s_cmp_lt_i32
// CHECK-NOT:     cmpi
// CHECK:         sop2 s_cselect_b32
// CHECK:         end_kernel
amdgcn.module @test_dead_cmpi_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_dead_cmpi_then_live {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    // Dead compare -- result unused, should be ignored by precondition check
    %dead = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    // Live compare -- consumed by select
    %live = lsir.cmpi i32 slt %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r = lsir.select %alloc2, %live, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    amdgcn.end_kernel
  }
}

// -----

// Overlapping i1 lifetimes: cmpi2 executes while cmpi1's result is still live.
// This would silently clobber SCC.

amdgcn.module @test_overlap_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_overlapping_i1 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    %cmp1 = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    // expected-error @+1 {{would clobber SCC from earlier compare; i1 lifetimes must not overlap}}
    %cmp2 = lsir.cmpi i32 slt %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %r1 = lsir.select %alloc2, %cmp1, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    amdgcn.end_kernel
  }
}

// -----

// Cross-block i1 usage: SCC is not preserved across block boundaries
// (any branch clobbers it).

amdgcn.module @test_crossblock_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @test_cross_block_i1 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %alloc0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %alloc1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %alloc2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %a = amdgcn.sop1 s_mov_b32 outs %alloc0 ins %c0 : !amdgcn.sgpr<0>, i32
    %b = amdgcn.sop1 s_mov_b32 outs %alloc1 ins %c10 : !amdgcn.sgpr<1>, i32
    // expected-error @+1 {{has consumer in a different block; SCC is not preserved across block boundaries}}
    %cmp = lsir.cmpi i32 eq %a, %b : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    cf.br ^bb1
  ^bb1:
    %r1 = lsir.select %alloc2, %cmp, %c42, %c99 : !amdgcn.sgpr<2>, i1, i32, i32
    amdgcn.end_kernel
  }
}
