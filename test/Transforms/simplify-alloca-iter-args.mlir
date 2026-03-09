// RUN: aster-opt %s --aster-simplify-alloca-iter-args --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @paired_basic_2stage
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32,
// CHECK:       scf.for {{.*}} iter_args(%[[ARG:.*]] = %{{.*}}) -> (memref<2xf32>)
// CHECK:         memref.store %[[V0]], %[[ARG]][%{{.*}}] : memref<2xf32>
// CHECK:         memref.store %[[V1]], %[[ARG]][%{{.*}}] : memref<2xf32>
// CHECK-NOT:     memref.load
// CHECK:         scf.yield
// Post-loop loads survive (forwarding is separate).
// CHECK:       memref.store
// CHECK:       memref.store
// CHECK:       memref.load
// CHECK:       memref.load
// CHECK:       return
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

// CHECK-LABEL: func.func @paired_four_elements
// CHECK:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (memref<4xf32>)
// CHECK-NOT:     memref.load
// CHECK:         scf.yield
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

// Multiple pairs (f32 + i32): 4 iter_args deduped to 2.

// CHECK-LABEL: func.func @paired_multiple_pairs
// CHECK:       scf.for {{.*}} iter_args({{.*}}) -> (memref<2xf32>, memref<2xi32>)
// CHECK-NOT:     memref.load
// CHECK:         scf.yield
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

// CHECK-LABEL: func.func @paired_rotation_3stage
// CHECK:       scf.for {{.*}} iter_args({{.*}}) -> (memref<2xf32>, memref<2xf32>)
// CHECK-NOT:     memref.load
// CHECK:         scf.yield
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
    memref.store %v0, %argSB[%c0] : memref<2xf32>
    memref.store %v1, %argSB[%c1] : memref<2xf32>
    %ld0 = memref.load %argDB[%c0] : memref<?xf32>
    %ld1 = memref.load %argDB[%c1] : memref<?xf32>
    %new_alloca = memref.alloca() : memref<2xf32>
    %new_cast = memref.cast %new_alloca : memref<2xf32> to memref<?xf32>
    scf.yield %new_alloca, %argSA, %new_cast, %argDA
        : memref<2xf32>, memref<2xf32>, memref<?xf32>, memref<?xf32>
  }
  memref.store %v0, %res#1[%c0] : memref<2xf32>
  memref.store %v1, %res#1[%c1] : memref<2xf32>
  %r0 = memref.load %res#3[%c0] : memref<?xf32>
  %r1 = memref.load %res#3[%c1] : memref<?xf32>
  return %r0, %r1 : f32, f32
}

// -----

// CHECK-LABEL: func.func @nonconstant_store_still_deduped
// CHECK:       scf.for {{.*}} iter_args(%[[ARG:.*]] = %{{.*}}) -> (memref<2xf32>)
// CHECK:         memref.store {{.*}}, %[[ARG]]
// CHECK:         memref.store {{.*}}, %[[ARG]]
// CHECK:         scf.yield
func.func @nonconstant_store_still_deduped(%v0: f32, %idx: index, %lb: index, %ub: index, %step: index) {
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
