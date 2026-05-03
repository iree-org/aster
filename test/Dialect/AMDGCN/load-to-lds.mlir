// RUN: aster-opt %s --verify-roundtrip

// Test G2S (Global-to-LDS / Direct-to-LDS) instructions.
// These are MUBUF loads with the LDS flag set. Data goes directly from
// global memory to LDS without touching VGPRs. M0 holds the LDS base offset.

// buffer_load_dword with LDS flag (32-bit G2S)
func.func @test_buffer_load_dword_lds(
    %m0: !amdgcn.m0<0>,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.buffer_load_lds_dword ins(%buf_desc, off_or_idx = %voffset, %soffset, %m0) args(%c0) {offen}
      : ins(!amdgcn.sgpr<[? + 4]>, off_or_idx = !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.m0<0>) args(i32) -> !amdgcn.read_token<flat>
  return
}

// buffer_load_dwordx4 with LDS flag (128-bit G2S)
func.func @test_buffer_load_dwordx4_lds(
    %m0: !amdgcn.m0<0>,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c64 = arith.constant 64 : i32
  %tok = amdgcn.buffer_load_lds_dwordx4 ins(%buf_desc, off_or_idx = %voffset, %soffset, %m0) args(%c64) {offen}
      : ins(!amdgcn.sgpr<[? + 4]>, off_or_idx = !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.m0<0>) args(i32) -> !amdgcn.read_token<flat>
  return
}

// G2S with inline-literal uniform offset (soffset = 0).
func.func @test_buffer_load_dword_lds_zero_soffset(
    %m0: !amdgcn.m0<0>,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.buffer_load_lds_dword ins(%buf_desc, off_or_idx = %voffset, %c0, %m0) args(%c0) {offen}
      : ins(!amdgcn.sgpr<[? + 4]>, off_or_idx = !amdgcn.vgpr, i32, !amdgcn.m0<0>) args(i32) -> !amdgcn.read_token<flat>
  return
}

// G2S without optional dynamic offset.
func.func @test_buffer_load_dword_lds_no_voffset(
    %m0: !amdgcn.m0<0>,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.buffer_load_lds_dword ins(%buf_desc, %soffset, %m0) args(%c0)
      : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.m0<0>) args(i32) -> !amdgcn.read_token<flat>
  return
}
