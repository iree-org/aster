// RUN: aster-opt %s --amdgcn-legalize-cf --split-input-file --verify-diagnostics | FileCheck %s

// Test: simple SCC conditional branch. falseDest (^bb2) is *not* next block
// (^bb1 is), so the pass chooses s_cbranch_scc0 to branch to ^bb2 when SCC=0,
// falling through to ^bb1 when SCC=1.

// CHECK-LABEL: kernel @test_cond_branch_scc
// CHECK:         s_cmp_lt_i32 outs(%[[SCC:.*]]) ins(%{{.*}}, %{{.*}}) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
// CHECK:         s_cbranch_scc0 %[[SCC]], true(^bb2) false(^bb1) : !amdgcn.scc<0>
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_scc target = <gfx942> {
  amdgcn.kernel @test_cond_branch_scc attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %s0 = alloca : !amdgcn.sgpr<0>
    %s1 = alloca : !amdgcn.sgpr<1>
    %scc = alloca : !amdgcn.scc<0>
    s_mov_b32 outs(%s0) ins(%c0) : outs(!amdgcn.sgpr<0>) ins(i32)
    s_mov_b32 outs(%s1) ins(%c10) : outs(!amdgcn.sgpr<1>) ins(i32)
    s_cmp_lt_i32 outs(%scc) ins(%s0, %s1) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
    lsir.cond_br %scc : !amdgcn.scc<0>, ^bb1, ^bb2
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}

// -----

// Test: unconditional branch. lsir.br -> s_branch.

// CHECK-LABEL: kernel @test_unconditional_branch
// CHECK:         s_branch ^bb1
// CHECK:       ^bb1:
// CHECK:         end_kernel
amdgcn.module @test_br target = <gfx942> {
  amdgcn.kernel @test_unconditional_branch attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    lsir.br ^bb1
  ^bb1:
    end_kernel
  }
}

// -----

// Test: VCC conditional branch. trueDest is next block (^bb1), so the pass
// uses s_cbranch_vccz to branch to ^bb2 when VCC=0, falling through to ^bb1.

// CHECK-LABEL: kernel @test_cond_branch_vcc
// CHECK:         v_cmp_lt_i32 outs(%[[VCC:.*]]) ins(%{{.*}}, %{{.*}})
// CHECK:         s_cbranch_vccz %[[VCC]], true(^bb2) false(^bb1)
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_vcc target = <gfx942> {
  amdgcn.kernel @test_cond_branch_vcc attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %c0 = arith.constant 0 : i32
    %v0 = alloca : !amdgcn.vgpr<0>
    %vcc_lo = alloca : !amdgcn.vcc_lo<0>
    %vcc_hi = alloca : !amdgcn.vcc_hi<0>
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    v_mov_b32 outs(%v0) ins(%c0) : outs(!amdgcn.vgpr<0>) ins(i32)
    v_cmp_lt_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, i32)
    lsir.cond_br %vcc : !amdgcn.vcc<0>, ^bb1, ^bb2
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}

// -----

// Test: SCC conditional branch where falseDest (^bb1) is the next block.
// This means the pass uses s_cbranch_scc1 to branch to ^bb2 when SCC=1,
// falling through to ^bb1 when SCC=0.

// CHECK-LABEL: kernel @test_false_is_fallthrough
// CHECK:         s_cmp_lt_i32
// CHECK:         s_cbranch_scc1 %{{.*}}, true(^bb2) false(^bb1)
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_false_fallthrough target = <gfx942> {
  amdgcn.kernel @test_false_is_fallthrough attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %s0 = alloca : !amdgcn.sgpr<0>
    %s1 = alloca : !amdgcn.sgpr<1>
    %scc = alloca : !amdgcn.scc<0>
    s_mov_b32 outs(%s0) ins(%c0) : outs(!amdgcn.sgpr<0>) ins(i32)
    s_mov_b32 outs(%s1) ins(%c10) : outs(!amdgcn.sgpr<1>) ins(i32)
    s_cmp_lt_i32 outs(%scc) ins(%s0, %s1) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
    lsir.cond_br %scc : !amdgcn.scc<0>, ^bb2, ^bb1
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}

// -----

// Test: loop with SCC back-edge. Entry: falseDest (^bb2) is not next (^bb1 is),
// so s_cbranch_scc0 is used. Back-edge: ^bb2 is the next block after ^bb1, so
// s_cbranch_scc1 is used to loop back to ^bb1.

// CHECK-LABEL: kernel @test_loop_scc
// CHECK:         s_cmp_gt_i32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}})
// CHECK:         s_cbranch_scc0 %{{.*}}, true(^bb2) false(^bb1)
// CHECK:       ^bb1:
// CHECK:         s_add_u32
// CHECK:         s_cmp_lt_i32
// CHECK:         s_cbranch_scc1 %{{.*}}, true(^bb1) false(^bb2)
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_loop target = <gfx942> {
  amdgcn.kernel @test_loop_scc arguments <[
        #amdgcn.buffer_arg<address_space = generic, access = read_only>,
        #amdgcn.buffer_arg<address_space = generic>
      ]>
      attributes {enable_workgroup_id_x = false, normal_forms = [#amdgcn.all_registers_allocated]}
  {
    %c8 = arith.constant 8 : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %s0 = alloca : !amdgcn.sgpr<0>
    %s1 = alloca : !amdgcn.sgpr<1>
    %s2 = alloca : !amdgcn.sgpr<2>
    %s3 = alloca : !amdgcn.sgpr<3>
    %s4 = alloca : !amdgcn.sgpr<4>
    %s5 = alloca : !amdgcn.sgpr<5>
    %s6 = alloca : !amdgcn.sgpr<6>
    %s7 = alloca : !amdgcn.sgpr<7>
    %s8 = alloca : !amdgcn.sgpr<8>
    %v0 = alloca : !amdgcn.vgpr<0>
    %scc = alloca : !amdgcn.scc<0>

    %ptr = make_register_range %s0, %s1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %dst1 = make_register_range %s2, %s3 : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %token = s_load_dwordx2 dest %dst1 addr %ptr offset c(%c0) : outs(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>) mods(i32) -> !amdgcn.read_token<constant>
    %dst2 = make_register_range %s4, %s5 : !amdgcn.sgpr<4>, !amdgcn.sgpr<5>
    %token_1 = s_load_dwordx2 dest %dst2 addr %ptr offset c(%c8) : outs(!amdgcn.sgpr<[4 : 6]>) ins(!amdgcn.sgpr<[0 : 2]>) mods(i32) -> !amdgcn.read_token<constant>
    s_waitcnt lgkmcnt = 0
    %c0_dup = arith.constant 0 : i32
    %token_3 = s_load_dword dest %s6 addr %dst1 offset c(%c0_dup) : outs(!amdgcn.sgpr<6>) ins(!amdgcn.sgpr<[2 : 4]>) mods(i32) -> !amdgcn.read_token<constant>
    s_waitcnt lgkmcnt = 0
    s_cmp_gt_i32 outs(%scc) ins(%s6, %c0) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<6>, i32)
    s_mov_b32 outs(%s7) ins(%c0) : outs(!amdgcn.sgpr<7>) ins(i32)
    lsir.cond_br %scc : !amdgcn.scc<0>, ^bb1, ^bb2
  ^bb1:
    s_lshl_b32 outs(%s8, %scc) ins(%s7, %c2) : outs(!amdgcn.sgpr<8>, !amdgcn.scc<0>) ins(!amdgcn.sgpr<7>, i32)
    v_mov_b32 outs(%v0) ins(%s8) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<8>)
    %wtok = global_store_dword data %v0 addr %dst2 offset d(%v0) + c(%c0_dup) : ins(!amdgcn.vgpr<0>, !amdgcn.sgpr<[4 : 6]>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.write_token<flat>
    s_add_u32 outs(%s7, %scc) ins(%s7, %c1) : outs(!amdgcn.sgpr<7>, !amdgcn.scc<0>) ins(!amdgcn.sgpr<7>, i32)
    s_cmp_lt_i32 outs(%scc) ins(%s7, %s6) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<7>, !amdgcn.sgpr<6>)
    lsir.cond_br %scc : !amdgcn.scc<0>, ^bb1, ^bb2
  ^bb2:
    end_kernel
  }
}

// -----

// Test: VCC conditional branch where falseDest (^bb1) is the next block.
// Uses s_cbranch_vccnz to branch to ^bb2 when VCC!=0, falling through to ^bb1.

// CHECK-LABEL: kernel @test_vcc_false_fallthrough
// CHECK:         v_cmp_lt_i32
// CHECK:         s_cbranch_vccnz %{{.*}}, true(^bb2) false(^bb1)
// CHECK:       ^bb1:
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         end_kernel
amdgcn.module @test_vcc_false_fallthrough target = <gfx942> {
  amdgcn.kernel @test_vcc_false_fallthrough attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %c0 = arith.constant 0 : i32
    %v0 = alloca : !amdgcn.vgpr<0>
    %vcc_lo = alloca : !amdgcn.vcc_lo<0>
    %vcc_hi = alloca : !amdgcn.vcc_hi<0>
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    v_mov_b32 outs(%v0) ins(%c0) : outs(!amdgcn.vgpr<0>) ins(i32)
    v_cmp_lt_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, i32)
    lsir.cond_br %vcc : !amdgcn.vcc<0>, ^bb2, ^bb1
  ^bb1:
    end_kernel
  ^bb2:
    end_kernel
  }
}
