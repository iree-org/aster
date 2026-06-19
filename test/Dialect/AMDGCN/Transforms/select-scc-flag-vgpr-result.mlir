// RUN: aster-opt %s -aster-to-amdgcn --split-input-file | FileCheck %s

// VGPR select with SCC condition -> v_cndmask + s_cselect_b64 broadcast.

// CHECK-LABEL: func.func @select_vgpr_result_scc_cond
// CHECK: amdgcn.s_cselect_b64 {{.*}} : outs(!amdgcn.vcc)
// CHECK: amdgcn.v_cndmask_b32 {{.*}} : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr, !amdgcn.vcc)
// CHECK-NOT: amdgcn.s_cselect_b32
func.func @select_vgpr_result_scc_cond(
    %dst: !amdgcn.vgpr,
    %tv: !amdgcn.vgpr,
    %lhs: !amdgcn.sgpr,
    %scc: !amdgcn.scc) -> !amdgcn.vgpr {
  %c8_i32 = arith.constant 8 : i32
  %c-1_i32 = arith.constant -1 : i32
  %cond = lsir.cmpi i32 ult %scc, %lhs, %c8_i32 : !amdgcn.scc, !amdgcn.sgpr, i32
  %result = lsir.select %dst, %cond, %tv, %c-1_i32 : !amdgcn.vgpr, !amdgcn.scc, !amdgcn.vgpr, i32
  return %result : !amdgcn.vgpr
}

// -----

// Scalar select still uses s_cselect_b32.

// CHECK-LABEL: func.func @select_sgpr_result_scc_cond
// CHECK: amdgcn.s_cselect_b32
// CHECK-NOT: amdgcn.v_cndmask_b32
func.func @select_sgpr_result_scc_cond(
    %dst: !amdgcn.sgpr,
    %tv: !amdgcn.sgpr,
    %fv: !amdgcn.sgpr,
    %lhs: !amdgcn.sgpr,
    %scc: !amdgcn.scc) -> !amdgcn.sgpr {
  %c8_i32 = arith.constant 8 : i32
  %cond = lsir.cmpi i32 ult %scc, %lhs, %c8_i32 : !amdgcn.scc, !amdgcn.sgpr, i32
  %result = lsir.select %dst, %cond, %tv, %fv : !amdgcn.sgpr, !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
  return %result : !amdgcn.sgpr
}

// -----

// Wave32: s_cselect_b32 vcc_lo broadcast + v_cndmask_b32.

// CHECK-LABEL: func.func @select_vgpr_result_scc_cond_wave32
// CHECK: amdgcn.s_cselect_b32 {{.*}} : outs(!amdgcn.vcc_lo)
// CHECK: amdgcn.v_cndmask_b32 {{.*}} : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr, !amdgcn.vcc_lo)
// CHECK-NOT: amdgcn.s_cselect_b64
amdgcn.module @select_wave32_mod target = #amdgcn.target<gfx1250> {
  func.func @select_vgpr_result_scc_cond_wave32(
      %dst: !amdgcn.vgpr,
      %tv: !amdgcn.vgpr,
      %lhs: !amdgcn.sgpr,
      %scc: !amdgcn.scc) -> !amdgcn.vgpr {
    %c8_i32 = arith.constant 8 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cond = lsir.cmpi i32 ult %scc, %lhs, %c8_i32 : !amdgcn.scc, !amdgcn.sgpr, i32
    %result = lsir.select %dst, %cond, %tv, %c-1_i32 : !amdgcn.vgpr, !amdgcn.scc, !amdgcn.vgpr, i32
    return %result : !amdgcn.vgpr
  }
}
