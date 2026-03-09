// RUN: aster-opt %s --aster-simplify-alloca-iter-args --aster-decompose-memref-iter-args --split-input-file | FileCheck %s

// Basic 2-stage: 2 iter_args (alloca + cast) deduped to 1 static iter_arg,
// in-loop loads forwarded to stored values, post-loop loads forwarded.
// Everything dead is cleaned up: stores, allocas, iter_args, constants.

// CHECK-LABEL: func.func @paired_basic_2stage
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32,
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return %[[V0]], %[[V1]] : f32, f32
func.func @paired_basic_2stage(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%c0] : memref<2xf32>
    memref.store %v1, %argS[%c1] : memref<2xf32>
    %ld0 = memref.load %argD[%c0] : memref<?xf32>
    %ld1 = memref.load %argD[%c1] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  memref.store %v0, %res#0[%c0] : memref<2xf32>
  memref.store %v1, %res#0[%c1] : memref<2xf32>
  %r0 = memref.load %res#1[%c0] : memref<?xf32>
  %r1 = memref.load %res#1[%c1] : memref<?xf32>
  return %r0, %r1 : f32, f32
}

// -----

// 4-element variant: same dedup + forwarding, verifies all 4 elements tracked.

// CHECK-LABEL: func.func @paired_four_elements
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32, %[[V2:.*]]: f32, %[[V3:.*]]: f32,
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return %[[V0]], %[[V1]], %[[V2]], %[[V3]] : f32, f32, f32, f32
func.func @paired_four_elements(%v0: f32, %v1: f32, %v2: f32, %v3: f32,
                                 %lb: index, %ub: index, %step: index) -> (f32, f32, f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %alloca = memref.alloca() : memref<4xf32>
  %cast = memref.cast %alloca : memref<4xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<4xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%c0] : memref<4xf32>
    memref.store %v1, %argS[%c1] : memref<4xf32>
    memref.store %v2, %argS[%c2] : memref<4xf32>
    memref.store %v3, %argS[%c3] : memref<4xf32>
    %ld0 = memref.load %argD[%c0] : memref<?xf32>
    %ld1 = memref.load %argD[%c1] : memref<?xf32>
    %ld2 = memref.load %argD[%c2] : memref<?xf32>
    %ld3 = memref.load %argD[%c3] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<4xf32>
    %new_cast = memref.cast %new_alloca : memref<4xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<4xf32>, memref<?xf32>
  }
  memref.store %v0, %res#0[%c0] : memref<4xf32>
  memref.store %v1, %res#0[%c1] : memref<4xf32>
  memref.store %v2, %res#0[%c2] : memref<4xf32>
  memref.store %v3, %res#0[%c3] : memref<4xf32>
  %r0 = memref.load %res#1[%c0] : memref<?xf32>
  %r1 = memref.load %res#1[%c1] : memref<?xf32>
  %r2 = memref.load %res#1[%c2] : memref<?xf32>
  %r3 = memref.load %res#1[%c3] : memref<?xf32>
  return %r0, %r1, %r2, %r3 : f32, f32, f32, f32
}

// -----

// Multiple pairs (f32 + i32): 4 iter_args deduped to 2, loads forwarded,
// everything cleaned up.

// CHECK-LABEL: func.func @paired_multiple_pairs
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return
func.func @paired_multiple_pairs(%v0: f32, %v1: f32, %w0: i32, %w1: i32,
                                  %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca_f = memref.alloca() : memref<2xf32>
  %cast_f = memref.cast %alloca_f : memref<2xf32> to memref<?xf32>
  %alloca_i = memref.alloca() : memref<2xi32>
  %cast_i = memref.cast %alloca_i : memref<2xi32> to memref<?xi32>
  %res:4 = scf.for %i = %lb to %ub step %step
      iter_args(%sfA = %alloca_f, %dfA = %cast_f,
                %siA = %alloca_i, %diA = %cast_i)
      -> (memref<2xf32>, memref<?xf32>, memref<2xi32>, memref<?xi32>) {
    memref.store %v0, %sfA[%c0] : memref<2xf32>
    memref.store %v1, %sfA[%c1] : memref<2xf32>
    %lf0 = memref.load %dfA[%c0] : memref<?xf32>
    %lf1 = memref.load %dfA[%c1] : memref<?xf32>
    memref.store %w0, %siA[%c0] : memref<2xi32>
    memref.store %w1, %siA[%c1] : memref<2xi32>
    %li0 = memref.load %diA[%c0] : memref<?xi32>
    %li1 = memref.load %diA[%c1] : memref<?xi32>
    %new_f = memref.alloca() : memref<2xf32>
    %new_fc = memref.cast %new_f : memref<2xf32> to memref<?xf32>
    %new_i = memref.alloca() : memref<2xi32>
    %new_ic = memref.cast %new_i : memref<2xi32> to memref<?xi32>
    scf.yield %new_f, %new_fc, %new_i, %new_ic
        : memref<2xf32>, memref<?xf32>, memref<2xi32>, memref<?xi32>
  }
  return
}

// -----

// Duplicate loads from the same index: all replaced with the stored value.

// CHECK-LABEL: func.func @paired_duplicate_loads
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32,
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return %[[V0]], %[[V1]] : f32, f32
func.func @paired_duplicate_loads(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%c0] : memref<2xf32>
    memref.store %v1, %argS[%c1] : memref<2xf32>
    %ld0a = memref.load %argD[%c0] : memref<?xf32>
    %ld0b = memref.load %argD[%c0] : memref<?xf32>
    %ld1 = memref.load %argD[%c1] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  return %v0, %v1 : f32, f32
}

// -----

// 3-stage rotation: paired iter_args deduped, stores forwarded to loads,
// rotation preserved (yield swaps A and B).

// CHECK-LABEL: func.func @paired_rotation_3stage
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32,
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return %[[V0]], %[[V1]] : f32, f32
func.func @paired_rotation_3stage(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca_a = memref.alloca() : memref<2xf32>
  %cast_a = memref.cast %alloca_a : memref<2xf32> to memref<?xf32>
  %alloca_b = memref.alloca() : memref<2xf32>
  %cast_b = memref.cast %alloca_b : memref<2xf32> to memref<?xf32>
  %res:4 = scf.for %i = %lb to %ub step %step
      iter_args(%argSA = %alloca_a, %argSB = %alloca_b,
                %argDA = %cast_a,   %argDB = %cast_b)
      -> (memref<2xf32>, memref<2xf32>, memref<?xf32>, memref<?xf32>) {
    // Stores to B, loads from B
    memref.store %v0, %argSB[%c0] : memref<2xf32>
    memref.store %v1, %argSB[%c1] : memref<2xf32>
    %ld0 = memref.load %argDB[%c0] : memref<?xf32>
    %ld1 = memref.load %argDB[%c1] : memref<?xf32>
    // Fresh alloca for A
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    // Rotation: A <- new, B <- old A (block args)
    scf.yield %new_alloca, %argSA, %new_cast, %argDA
        : memref<2xf32>, memref<2xf32>, memref<?xf32>, memref<?xf32>
  }
  // Post-loop forwarding on B results
  memref.store %v0, %res#1[%c0] : memref<2xf32>
  memref.store %v1, %res#1[%c1] : memref<2xf32>
  %r0 = memref.load %res#3[%c0] : memref<?xf32>
  %r1 = memref.load %res#3[%c1] : memref<?xf32>
  return %r0, %r1 : f32, f32
}

// -----

//===----------------------------------------------------------------------===//
// Unpaired static alloca self-forwarding
//===----------------------------------------------------------------------===//

// Unpaired iter_arg: stores and loads target the same block arg (no cast).
// In-loop and post-loop stores+loads are forwarded. Everything cleaned up.

// CHECK-LABEL: func.func @unpaired_self_forwarding
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32,
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return %[[V0]], %[[V1]] : f32, f32
func.func @unpaired_self_forwarding(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  memref.store %v0, %alloca[%c0] : memref<2xf32>
  memref.store %v1, %alloca[%c1] : memref<2xf32>
  %res = scf.for %i = %lb to %ub step %step
      iter_args(%buf = %alloca) -> (memref<2xf32>) {
    memref.store %v0, %buf[%c0] : memref<2xf32>
    memref.store %v1, %buf[%c1] : memref<2xf32>
    %ld0 = memref.load %buf[%c0] : memref<2xf32>
    %ld1 = memref.load %buf[%c1] : memref<2xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    scf.yield %new_alloca : memref<2xf32>
  }
  memref.store %v0, %res[%c0] : memref<2xf32>
  memref.store %v1, %res[%c1] : memref<2xf32>
  %r0 = memref.load %res[%c0] : memref<2xf32>
  %r1 = memref.load %res[%c1] : memref<2xf32>
  return %r0, %r1 : f32, f32
}

// -----

// Unpaired with dead post-loop stores only (no post-loop loads).
// In-loop loads forwarded, post-loop stores erased, everything cleaned up.

// CHECK-LABEL: func.func @unpaired_dead_post_loop_stores
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return
func.func @unpaired_dead_post_loop_stores(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %res = scf.for %i = %lb to %ub step %step
      iter_args(%buf = %alloca) -> (memref<2xf32>) {
    memref.store %v0, %buf[%c0] : memref<2xf32>
    memref.store %v1, %buf[%c1] : memref<2xf32>
    %ld0 = memref.load %buf[%c0] : memref<2xf32>
    %ld1 = memref.load %buf[%c1] : memref<2xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    scf.yield %new_alloca : memref<2xf32>
  }
  memref.store %v0, %res[%c0] : memref<2xf32>
  memref.store %v1, %res[%c1] : memref<2xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// Negative tests: forwarding bails out but cast folding still fires.
//===----------------------------------------------------------------------===//

// Non-constant store index: forwarding bails out.
// SimplifyAllocaIterArgs deduplicates (1 iter_arg instead of 2).
// Stores and iter_arg remain because the dynamic index prevents forwarding.

// CHECK-LABEL: func.func @negative_nonconstant_store
// CHECK-SAME:    (%[[V0:[^:]*]]: f32, %[[IDX:[^:]*]]: index,
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<2xf32>
// CHECK:         scf.for {{.*}} iter_args(%[[BUF:.*]] = %[[ALLOCA]]) -> (memref<2xf32>)
// CHECK:           memref.store %[[V0]], %[[BUF]][%[[IDX]]] : memref<2xf32>
// CHECK:           memref.store %[[V0]], %[[BUF]][%{{.*}}] : memref<2xf32>
// CHECK:           scf.yield %{{.*}} : memref<2xf32>
func.func @negative_nonconstant_store(%v0: f32, %idx: index, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%idx] : memref<2xf32>
    memref.store %v0, %argS[%c0] : memref<2xf32>
    %ld0 = memref.load %argD[%c0] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  return
}

// -----

// Load before store (dominance violation): forwarding bails out.
// After dedup, loads become unused (no post-loop consumers), stores also dead.
// Canonicalization removes everything.

// CHECK-LABEL: func.func @negative_load_before_store
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return
func.func @negative_load_before_store(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    %ld0 = memref.load %argD[%c0] : memref<?xf32>
    %ld1 = memref.load %argD[%c1] : memref<?xf32>
    memref.store %v0, %argS[%c0] : memref<2xf32>
    memref.store %v1, %argS[%c1] : memref<2xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  return
}

// -----

// Multiple stores to same index: forwarding bails out.
// Stores and iter_arg remain because duplicate index prevents forwarding.

// CHECK-LABEL: func.func @negative_duplicate_store_index
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<2xf32>
// CHECK:         scf.for {{.*}} iter_args(%[[BUF:.*]] = %[[ALLOCA]]) -> (memref<2xf32>)
// CHECK:           memref.store {{.*}}, %[[BUF]][%{{.*}}] : memref<2xf32>
// CHECK:           memref.store {{.*}}, %[[BUF]][%{{.*}}] : memref<2xf32>
// CHECK:           scf.yield %{{.*}} : memref<2xf32>
func.func @negative_duplicate_store_index(%v0: f32, %v1: f32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%c0] : memref<2xf32>
    memref.store %v1, %argS[%c0] : memref<2xf32>
    %ld0 = memref.load %argD[%c0] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  return
}

// -----

// Load at index with no corresponding store: forwarding bails out.
// After dedup, load becomes unused (no post-loop consumers), store also dead.
// Canonicalization removes everything.

// CHECK-LABEL: func.func @negative_missing_store_for_load
// CHECK-NOT:     memref
// CHECK-NOT:     scf.for
// CHECK:         return
func.func @negative_missing_store_for_load(%v0: f32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<2xf32>
  %cast = memref.cast %alloca : memref<2xf32> to memref<?xf32>
  %res:2 = scf.for %i = %lb to %ub step %step
      iter_args(%argS = %alloca, %argD = %cast) -> (memref<2xf32>, memref<?xf32>) {
    memref.store %v0, %argS[%c0] : memref<2xf32>
    %ld1 = memref.load %argD[%c1] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %new_cast : memref<2xf32>, memref<?xf32>
  }
  return
}
