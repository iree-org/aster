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
