// RUN: aster-opt %s --pass-pipeline='builtin.module(canonicalize,amdgcn.module(amdgcn-convert-waits))' | FileCheck %s

// gfx1250 has no plain s_barrier: cross_wave_token_barrier lowers to the split
// s_barrier_signal -1 / s_barrier_wait -1 pair. CDNA3 keeps a single s_barrier.

// CHECK-LABEL:   func.func @gfx1250_cross_wave_barrier(
//       CHECK:     amdgcn.s_barrier_signal -1
//  CHECK-NEXT:     amdgcn.s_barrier_wait -1
//   CHECK-NOT:     cross_wave_token_barrier
amdgcn.module @convert_waits_gfx1250_mod target = #amdgcn.target<gfx1250> {
func.func @gfx1250_cross_wave_barrier(%addr: !amdgcn.vgpr, %data: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %wf = amdgcn.wait_gfx1250 deps %token : !amdgcn.write_token<shared> -> !amdgcn.fence_token
  %fence = amdgcn.cross_wave_token_barrier deps %wf : !amdgcn.fence_token
  return
}
}

// CHECK-LABEL:   func.func @cdna3_cross_wave_barrier(
//   CHECK-NOT:     s_barrier_signal
//   CHECK-NOT:     s_barrier_wait
//       CHECK:     amdgcn.s_barrier
//   CHECK-NOT:     cross_wave_token_barrier
amdgcn.module @convert_waits_cdna3_mod target = <gfx942> {
func.func @cdna3_cross_wave_barrier(%addr: !amdgcn.vgpr, %data: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!amdgcn.vgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<shared>
  %wf = amdgcn.wait deps %token : !amdgcn.write_token<shared> -> !amdgcn.fence_token
  %fence = amdgcn.cross_wave_token_barrier deps %wf : !amdgcn.fence_token
  return
}
}
