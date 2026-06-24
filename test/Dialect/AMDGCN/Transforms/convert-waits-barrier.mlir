// RUN: aster-opt %s --pass-pipeline='builtin.module(canonicalize,amdgcn.module(amdgcn-convert-waits))' | FileCheck %s

// amdgcn.barrier is the target-neutral workgroup barrier. convert-waits lowers
// it late: the split s_barrier_signal -1 / s_barrier_wait -1 pair on gfx1250,
// and a single s_barrier on CDNA3/CDNA4.

// CHECK-LABEL:   func.func @gfx1250_barrier(
//       CHECK:     amdgcn.s_barrier_signal -1
//  CHECK-NEXT:     amdgcn.s_barrier_wait -1
//   CHECK-NOT:     amdgcn.barrier
amdgcn.module @convert_waits_barrier_gfx1250_mod target = #amdgcn.target<gfx1250> {
func.func @gfx1250_barrier() {
  amdgcn.barrier
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
  amdgcn.barrier
  return
}
}
