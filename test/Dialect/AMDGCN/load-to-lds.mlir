// RUN: aster-opt %s --verify-roundtrip

// Test G2S (Global-to-LDS / Direct-to-LDS) instructions.
// These are MUBUF loads with the LDS flag set. Data goes directly from
// global memory to LDS without touching VGPRs. M0 holds the LDS base offset.

// buffer_load_dword with LDS flag (32-bit G2S)
func.func @test_buffer_load_dword_lds(
    %m0: !amdgcn.m0,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.load_lds buffer_load_dword_lds m0 %m0 addr %buf_desc
      offset u(%soffset) + d(%voffset) + c(%c0)
      : ins(!amdgcn.m0, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

// buffer_load_dwordx4 with LDS flag (128-bit G2S)
func.func @test_buffer_load_dwordx4_lds(
    %m0: !amdgcn.m0,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr,
    %voffset: !amdgcn.vgpr) {
  %c64 = arith.constant 64 : i32
  %tok = amdgcn.load_lds buffer_load_dwordx4_lds m0 %m0 addr %buf_desc
      offset u(%soffset) + d(%voffset) + c(%c64)
      : ins(!amdgcn.m0, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

// G2S without optional uniform offset
func.func @test_buffer_load_dword_lds_no_soffset(
    %m0: !amdgcn.m0,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %voffset: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.load_lds buffer_load_dword_lds m0 %m0 addr %buf_desc
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.m0, !amdgcn.sgpr<[? + 4]>, !amdgcn.vgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}

// G2S without optional dynamic offset
func.func @test_buffer_load_dword_lds_no_voffset(
    %m0: !amdgcn.m0,
    %buf_desc: !amdgcn.sgpr<[? + 4]>,
    %soffset: !amdgcn.sgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.load_lds buffer_load_dword_lds m0 %m0 addr %buf_desc
      offset u(%soffset) + c(%c0)
      : ins(!amdgcn.m0, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, i32)
      -> !amdgcn.write_token<flat>
  return
}
