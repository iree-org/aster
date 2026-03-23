// Copyright 2026 The ASTER Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

// RUN: aster-opt %s --split-input-file \
// RUN:   --aster-scf-pipeline="rotate-kernel=true" | FileCheck %s

!vgpr = !amdgcn.vgpr

// 2-stage pipeline with sched.rotate_head on the compute op.
// After pipelining, the kernel body has load(K+1) then compute(K).
// rotate-kernel peels so compute fires first:
//   prologue:  load(lb)                  [peeled rest]
//   kernel:    compute(K), load(K+step)  [rotated body]
//   epilogue:  compute(ub-step)          [peeled head]
//
// CHECK-LABEL: func.func @rotate_2stage
// Peeled prologue: load cloned before loop.
//     CHECK:   amdgcn.test_inst{{.*}}{load}
// Rotated loop with additional iter_arg for crossing value.
//     CHECK:   scf.for
// Compute first (head).
//     CHECK:     amdgcn.test_inst{{.*}}{compute, sched.rotate_head}
// Load second (rest, shifted IV).
//     CHECK:     arith.addi
//     CHECK:     amdgcn.test_inst{{.*}}{load}
//     CHECK:     scf.yield
// Peeled epilogue: compute after loop.
//     CHECK:   amdgcn.test_inst{{.*}}{compute, sched.rotate_head}

func.func @rotate_2stage(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr
  %s1 = amdgcn.alloca : !vgpr
  %s_out = amdgcn.alloca : !vgpr
  %init = amdgcn.test_inst outs %s0 : (!vgpr) -> !vgpr

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr {
    %data = amdgcn.test_inst outs %s1
        {load, sched.stage = 0 : i32} : (!vgpr) -> !vgpr

    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %data
        {compute, sched.stage = 1 : i32, sched.rotate_head}
        : (!vgpr, !vgpr, !vgpr) -> !vgpr

    scf.yield %new_acc : !vgpr
  }
  return
}

// -----

// No sched.rotate_head: rotate-kernel is a no-op.
// CHECK-LABEL: func.func @no_rotate_head
// Pipeline prologue.
//     CHECK:   amdgcn.test_inst{{.*}}{load}
// Kernel loop: normal order (load first, compute second).
//     CHECK:   scf.for
//     CHECK:     amdgcn.test_inst{{.*}}{load}
//     CHECK:     amdgcn.test_inst{{.*}}{compute}
//     CHECK:     scf.yield
// Epilogue.
//     CHECK:   amdgcn.test_inst{{.*}}{compute}

!vgpr_2 = !amdgcn.vgpr

func.func @no_rotate_head(%ub: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %s0 = amdgcn.alloca : !vgpr_2
  %s1 = amdgcn.alloca : !vgpr_2
  %s_out = amdgcn.alloca : !vgpr_2
  %init = amdgcn.test_inst outs %s0 : (!vgpr_2) -> !vgpr_2

  scf.for %k = %c0 to %ub step %c1 iter_args(%acc = %init) -> !vgpr_2 {
    %data = amdgcn.test_inst outs %s1
        {load, sched.stage = 0 : i32} : (!vgpr_2) -> !vgpr_2

    %new_acc = amdgcn.test_inst outs %s_out ins %acc, %data
        {compute, sched.stage = 1 : i32}
        : (!vgpr_2, !vgpr_2, !vgpr_2) -> !vgpr_2

    scf.yield %new_acc : !vgpr_2
  }
  return
}
