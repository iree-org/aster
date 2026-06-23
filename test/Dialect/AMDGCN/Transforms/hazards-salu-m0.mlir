// RUN: aster-opt %s --amdgcn-hazards --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_salu_m0_buffer_load_lds(
// CHECK:           amdgcn.s_mov_b32 outs(%{{.*}}) ins(%{{.*}})
// CHECK:           amdgcn.s_nop 0
// CHECK:           amdgcn.buffer_load_lds_dword
// CHECK:           return
func.func @test_salu_m0_buffer_load_lds(
    %m0: !amdgcn.m0<0>,
    %rsrc: !amdgcn.sgpr<[0 : 4]>,
    %soff: !amdgcn.sgpr<0>,
    %voff: !amdgcn.vgpr<0>) {
  %c44 = arith.constant 44 : i32
  %c0 = arith.constant 0 : i32
  amdgcn.s_mov_b32 outs(%m0) ins(%c44) : outs(!amdgcn.m0<0>) ins(i32)
  %tok = amdgcn.buffer_load_lds_dword addr %rsrc m0 %m0 offset u(%soff) + off_idx(%voff) + c(%c0) {offen} : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.m0<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.read_token<flat>
  return
}

// -----
// CHECK-LABEL:   func.func @test_salu_m0_no_hazard(
// CHECK:           amdgcn.s_mov_b32 outs(%{{.*}}) ins(%{{.*}})
// CHECK:           amdgcn.s_mov_b32
// CHECK-NOT:       amdgcn.s_nop
// CHECK:           amdgcn.buffer_load_lds_dword
// CHECK:           return
func.func @test_salu_m0_no_hazard(
    %m0: !amdgcn.m0<0>,
    %rsrc: !amdgcn.sgpr<[0 : 4]>,
    %soff: !amdgcn.sgpr<0>,
    %voff: !amdgcn.vgpr<0>,
    %tmp: !amdgcn.sgpr<1>) {
  %c44 = arith.constant 44 : i32
  %c0 = arith.constant 0 : i32
  amdgcn.s_mov_b32 outs(%m0) ins(%c44) : outs(!amdgcn.m0<0>) ins(i32)
  // Intervening SALU satisfies the 1 wait state; no NOP before buffer_load_lds_dword.
  amdgcn.s_mov_b32 outs(%tmp) ins(%c44) : outs(!amdgcn.sgpr<1>) ins(i32)
  %tok = amdgcn.buffer_load_lds_dword addr %rsrc m0 %m0 offset u(%soff) + off_idx(%voff) + c(%c0) {offen} : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.m0<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.read_token<flat>
  return
}
