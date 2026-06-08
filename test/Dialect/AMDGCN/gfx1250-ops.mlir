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
    %d3: !amdgcn.sgpr<[? + 4]>) -> !amdgcn.read_token<flat> {
  %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
      : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 8]>, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 4]>) -> !amdgcn.read_token<flat>
  return %tok : !amdgcn.read_token<flat>
}

//===----------------------------------------------------------------------===//
// SOPP control ops (s_set_vgpr_msb, s_setprio_inc_wg)
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
