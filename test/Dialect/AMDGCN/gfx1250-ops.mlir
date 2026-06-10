// RUN: aster-opt %s --verify-roundtrip
// Roundtrip coverage for all gfx1250 (GFX12.5) instructions.

// vx8 = 8-VGPR range (v8f32 accumulator, or 16xf16/bf16 packed in 8 VGPRs).
!vx8 = !amdgcn.vgpr<[? + 8]>

//===----------------------------------------------------------------------===//
// DS loads (ds_load_b128 -- gfx1250 plain DS load, opcode 0x0ff)
//===----------------------------------------------------------------------===//

func.func @test_ds_load_b128(%addr: !amdgcn.vgpr, %dst4: !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]> {
  %offset = arith.constant 0 : i32
  %result, %tok = amdgcn.ds_load_b128 dest %dst4 addr %addr offset c(%offset) : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  return %result : !amdgcn.vgpr<[? + 4]>
}

//===----------------------------------------------------------------------===//
// VIMAGE tensor load (tensor_load_to_lds -- 4 SGPR descriptors, dest-less)
//===----------------------------------------------------------------------===//

func.func @test_tensor_load_to_lds(
    %d0: !amdgcn.sgpr<[? + 4]>,
    %d1: !amdgcn.sgpr<[? + 8]>,
    %d2: !amdgcn.sgpr<[? + 4]>,
    %d3: !amdgcn.sgpr<[? + 4]>) -> !amdgcn.read_token<tensor> {
  %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
      : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 8]>, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 4]>) -> !amdgcn.read_token<tensor>
  return %tok : !amdgcn.read_token<tensor>
}

//===----------------------------------------------------------------------===//
// SOPP control + wait ops (s_set_vgpr_msb, s_setprio_inc_wg, s_wait_*cnt)
//===----------------------------------------------------------------------===//

func.func @test_s_set_vgpr_msb_0() {
  amdgcn.s_set_vgpr_msb 0
  return
}

func.func @test_s_set_vgpr_msb_nonzero() {
  amdgcn.s_set_vgpr_msb 3
  return
}

func.func @test_s_setprio_inc_wg() {
  amdgcn.s_setprio_inc_wg 3
  amdgcn.s_setprio_inc_wg 0
  return
}

func.func @test_s_wait_counters() {
  amdgcn.s_wait_loadcnt 0
  amdgcn.s_wait_storecnt 1
  amdgcn.s_wait_dscnt 2
  amdgcn.s_wait_kmcnt 3
  amdgcn.s_wait_tensorcnt 4
  amdgcn.s_wait_asynccnt 5
  amdgcn.s_wait_xcnt 6
  return
}

//===----------------------------------------------------------------------===//
// Workgroup-cluster id ops (cluster_id, cluster_workgroup_*)
//===----------------------------------------------------------------------===//

func.func @test_cluster_id_dims() -> !amdgcn.sgpr {
  %x = amdgcn.cluster_id x : !amdgcn.sgpr
  %y = amdgcn.cluster_id y : !amdgcn.sgpr
  %z = amdgcn.cluster_id z : !amdgcn.sgpr
  return %x : !amdgcn.sgpr
}

func.func @test_cluster_workgroup_id_dims() -> !amdgcn.sgpr {
  %x = amdgcn.cluster_workgroup_id x : !amdgcn.sgpr
  %y = amdgcn.cluster_workgroup_id y : !amdgcn.sgpr
  %z = amdgcn.cluster_workgroup_id z : !amdgcn.sgpr
  return %x : !amdgcn.sgpr
}

func.func @test_cluster_workgroup_max_id_dims() -> !amdgcn.sgpr {
  %x = amdgcn.cluster_workgroup_max_id x : !amdgcn.sgpr
  %y = amdgcn.cluster_workgroup_max_id y : !amdgcn.sgpr
  %z = amdgcn.cluster_workgroup_max_id z : !amdgcn.sgpr
  return %x : !amdgcn.sgpr
}

//===----------------------------------------------------------------------===//
// gfx1250 split wait op (amdgcn.wait_gfx1250; disjoint from the CDNA amdgcn.wait)
//===----------------------------------------------------------------------===//

func.func @test_s_wait_fused_counters() {
  amdgcn.s_wait_loadcnt_dscnt 259
  amdgcn.s_wait_storecnt_dscnt 0
  return
}

// s_wait_alu is the ALU register-forwarding-hazard wait (depctr), not a memory
// wait counter.
func.func @test_s_wait_alu() {
  amdgcn.s_wait_alu 0
  return
}

func.func @test_s_barrier_gfx1250() {
  amdgcn.s_barrier_signal -1
  amdgcn.s_barrier_wait -1
  return
}

func.func @test_wait_gfx1250_explicit() {
  %wf0 = amdgcn.wait_gfx1250 load_cnt 1 store_cnt 2 ds_cnt 3 km_cnt 4 tensor_cnt 5 -> !amdgcn.fence_token
  return
}

func.func @test_wait_gfx1250_token(%d0: !amdgcn.sgpr<[? + 4]>,
    %d1: !amdgcn.sgpr<[? + 8]>, %d2: !amdgcn.sgpr<[? + 4]>,
    %d3: !amdgcn.sgpr<[? + 4]>) {
  %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
      : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 8]>, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 4]>) -> !amdgcn.read_token<tensor>
  %wf1 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
  return
}

//===----------------------------------------------------------------------===//
// WMMA (v_wmma_f32_16x16x32_{f16,bf16}; optional matrix_*_reuse / neg modifiers,
// inline-f32 C accumulator)
//===----------------------------------------------------------------------===//

func.func @test_wmma_f16(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

func.func @test_wmma_bf16(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_bf16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

// C accumulator (src2) may be an inline f32 constant (LLVM VISrc), not just a VGPR.
func.func @test_wmma_f16_inline_c(%dst: !vx8, %a: !vx8, %b: !vx8) -> !vx8 {
  %c = arith.constant 0.0 : f32
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      : outs(!vx8) ins(!vx8, !vx8, f32)
  return %result : !vx8
}

func.func @test_wmma_f16_reuse(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      matrix_a_reuse(unit) matrix_b_reuse(unit)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

func.func @test_wmma_f16_neg(%dst: !vx8, %a: !vx8, %b: !vx8, %c: !vx8) -> !vx8 {
  %result = amdgcn.v_wmma_f32_16x16x32_f16 outs(%dst) ins(%a, %b, %c)
      neg_lo(array<i1: false, false, true>) neg_hi(array<i1: false, false, true>)
      : outs(!vx8) ins(!vx8, !vx8, !vx8)
  return %result : !vx8
}

//===----------------------------------------------------------------------===//
// Modules and libraries (gfx1250 / gfx1251 target + gfx12_50 isa parse)
//===----------------------------------------------------------------------===//

amdgcn.module @gfx1250_module target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @gfx1250_kernel {
    amdgcn.end_kernel
  }
}

amdgcn.module @gfx1251_module target = #amdgcn.target<gfx1251> {
  amdgcn.kernel @gfx1251_kernel {
    amdgcn.end_kernel
  }
}

amdgcn.library @library_multi_isa_gfx12_50 isa = [#amdgcn.isa<cdna3>, #amdgcn.isa<gfx12_50>] {
  func.func @multi_target_func_gfx12_50() {
    return
  }
}

//===----------------------------------------------------------------------===//
// v_cndmask_b32 with VCC_LO mask
//===----------------------------------------------------------------------===//

func.func @test_v_cndmask_b32_vcc_lo(
    %dst: !amdgcn.vgpr,
    %src0: !amdgcn.vgpr,
    %src1: !amdgcn.vgpr,
    %mask: !amdgcn.vcc_lo) -> !amdgcn.vgpr {
  %result = amdgcn.v_cndmask_b32 outs(%dst) ins(%src0, %src1, %mask)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vcc_lo)
  return %result : !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// v_cmp + s_cbranch with VCC_LO
//===----------------------------------------------------------------------===//

func.func @test_v_cmp_lt_i32_vcc_lo(
    %dst: !amdgcn.vcc_lo,
    %lhs: !amdgcn.vgpr,
    %rhs: !amdgcn.vgpr) -> !amdgcn.vcc_lo {
  %result = amdgcn.v_cmp_lt_i32 outs(%dst) ins(%lhs, %rhs)
      : outs(!amdgcn.vcc_lo) ins(!amdgcn.vgpr, !amdgcn.vgpr)
  return %result : !amdgcn.vcc_lo
}

func.func @test_s_cbranch_vccnz_vcc_lo(%cond: !amdgcn.vcc_lo) {
  amdgcn.s_cbranch_vccnz %cond, true(^taken) false(^fallthru) : !amdgcn.vcc_lo
^fallthru:
  return
^taken:
  return
}
