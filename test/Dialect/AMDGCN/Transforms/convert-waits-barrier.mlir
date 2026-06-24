// RUN: aster-opt %s --pass-pipeline='builtin.module(canonicalize,amdgcn.module(amdgcn-convert-waits))' | FileCheck %s

// CHECK-LABEL:   func.func @gfx1250_barrier(
//       CHECK:     amdgcn.s_barrier_signal -1
//  CHECK-NEXT:     amdgcn.s_barrier_wait -1
//   CHECK-NOT:     amdgcn.barrier
amdgcn.module @convert_waits_barrier_gfx1250_mod target = #amdgcn.target<gfx1250> {
func.func @gfx1250_barrier() {
  amdgcn.barrier scope(<workgroup>)
  return
}
}

// CHECK-LABEL:   func.func @cdna3_barrier(
//   CHECK-NOT:     s_barrier_signal
//   CHECK-NOT:     s_barrier_wait
//       CHECK:     amdgcn.s_barrier
//   CHECK-NOT:     amdgcn.barrier
amdgcn.module @convert_waits_barrier_cdna3_mod target = <gfx942> {
func.func @cdna3_barrier() {
  amdgcn.barrier scope(<workgroup>)
  return
}
}
