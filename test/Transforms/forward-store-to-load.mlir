// RUN: aster-opt %s --aster-forward-store-to-load | FileCheck %s

// -----

// CHECK-LABEL: func.func @basic_forwarding
// CHECK-SAME:    (%[[VAL:.*]]: f32)
// CHECK-NEXT:    return %[[VAL]] : f32
func.func @basic_forwarding(%val: f32) -> f32 {
  %alloca = memref.alloca() : memref<f32>
  memref.store %val, %alloca[] : memref<f32>
  %loaded = memref.load %alloca[] : memref<f32>
  return %loaded : f32
}

// -----

// CHECK-LABEL: func.func @chain_forwarding
// CHECK-SAME:    (%[[V1:.*]]: f32, %[[V2:.*]]: f32)
// CHECK-NEXT:    return %[[V1]], %[[V2]] : f32, f32
func.func @chain_forwarding(%v1: f32, %v2: f32) -> (f32, f32) {
  %alloca = memref.alloca() : memref<f32>
  memref.store %v1, %alloca[] : memref<f32>
  %load1 = memref.load %alloca[] : memref<f32>
  memref.store %v2, %alloca[] : memref<f32>
  %load2 = memref.load %alloca[] : memref<f32>
  return %load1, %load2 : f32, f32
}

// -----

// CHECK-LABEL: func.func @interleaved_allocas
// CHECK-SAME:    (%[[VA:.*]]: f32, %[[VB:.*]]: f32)
// CHECK-NEXT:    return %[[VA]], %[[VB]] : f32, f32
func.func @interleaved_allocas(%va: f32, %vb: f32) -> (f32, f32) {
  %alloca_a = memref.alloca() : memref<f32>
  memref.store %va, %alloca_a[] : memref<f32>
  %alloca_b = memref.alloca() : memref<f32>
  memref.store %vb, %alloca_b[] : memref<f32>
  %load_a = memref.load %alloca_a[] : memref<f32>
  %load_b = memref.load %alloca_b[] : memref<f32>
  return %load_a, %load_b : f32, f32
}

// -----

// CHECK-LABEL: func.func @dead_store_elimination
// CHECK-SAME:    (%[[V1:.*]]: f32, %[[V2:.*]]: f32)
// CHECK-NEXT:    return
func.func @dead_store_elimination(%v1: f32, %v2: f32) {
  %alloca = memref.alloca() : memref<f32>
  memref.store %v1, %alloca[] : memref<f32>
  memref.store %v2, %alloca[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func.func @no_cross_block
// CHECK:         memref.store
// CHECK:         memref.load
func.func @no_cross_block(%val: f32, %cond: i1) -> f32 {
  %alloca = memref.alloca() : memref<f32>
  memref.store %val, %alloca[] : memref<f32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %loaded = memref.load %alloca[] : memref<f32>
  return %loaded : f32
^bb2:
  %zero = arith.constant 0.0 : f32
  return %zero : f32
}

// -----

// Indexed memref forwarding: rank-1 with static shape and constant indices.
// CHECK-LABEL: func.func @indexed_forwarding
// CHECK-SAME:    (%[[VAL:.*]]: f32)
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    return %[[VAL]] : f32
func.func @indexed_forwarding(%val: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<4xf32>
  memref.store %val, %alloca[%c0] : memref<4xf32>
  %loaded = memref.load %alloca[%c0] : memref<4xf32>
  return %loaded : f32
}

// -----

// CHECK-LABEL: func.func @rank1_scalar
// CHECK-SAME:    (%[[VAL:.*]]: f32)
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    return %[[VAL]] : f32
func.func @rank1_scalar(%val: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<1xf32>
  memref.store %val, %alloca[%c0] : memref<1xf32>
  %loaded = memref.load %alloca[%c0] : memref<1xf32>
  return %loaded : f32
}

// -----

// CHECK-LABEL: func.func @no_forwarding_across_loop
// CHECK-SAME:    (%[[INIT:.*]]: f32)
// CHECK:         memref.store %[[INIT]], %[[ALLOCA:.*]][] : memref<f32>
// CHECK:         scf.for
// CHECK:         }
// CHECK-NEXT:    %[[POST:.*]] = memref.load %[[ALLOCA]][] : memref<f32>
// CHECK-NEXT:    return %[[POST]]
func.func @no_forwarding_across_loop(%init: f32) -> f32 {
  %alloca = memref.alloca() : memref<f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  memref.store %init, %alloca[] : memref<f32>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %alloca[] : memref<f32>
    %inc = arith.addf %v, %v : f32
    memref.store %inc, %alloca[] : memref<f32>
  }
  %result = memref.load %alloca[] : memref<f32>
  return %result : f32
}

// -----

// CHECK-LABEL: func.func @mfma_accumulator_kt2
// CHECK-SAME:    (%[[INIT:.*]]: f32, %[[A0:.*]]: f32, %[[A1:.*]]: f32)
// CHECK-NEXT:    %[[R0:.*]] = arith.addf %[[INIT]], %[[A0]] : f32
// CHECK-NEXT:    %[[R1:.*]] = arith.addf %[[R0]], %[[A1]] : f32
// CHECK-NEXT:    return %[[R1]] : f32
func.func @mfma_accumulator_kt2(%init: f32, %a0: f32, %a1: f32) -> f32 {
  %alloca = memref.alloca() : memref<f32>
  // kt=0: store init, load, compute, store result
  memref.store %init, %alloca[] : memref<f32>
  %c0_load = memref.load %alloca[] : memref<f32>
  %r0 = arith.addf %c0_load, %a0 : f32
  memref.store %r0, %alloca[] : memref<f32>
  // kt=1: load (redundant), compute, store result
  %c1_load = memref.load %alloca[] : memref<f32>
  %r1 = arith.addf %c1_load, %a1 : f32
  memref.store %r1, %alloca[] : memref<f32>
  // Post-loop load
  %final = memref.load %alloca[] : memref<f32>
  return %final : f32
}

// -----

// Multi-index forwarding: stores and loads to different constant indices
// in a rank-1 memref with static shape.
// CHECK-LABEL: func.func @multi_index_forwarding
// CHECK-SAME:    (%[[V0:.*]]: f32, %[[V1:.*]]: f32, %[[V2:.*]]: f32)
// CHECK-NOT:     memref.store
// CHECK-NOT:     memref.load
// CHECK:         return %[[V0]], %[[V1]], %[[V2]] : f32, f32, f32
func.func @multi_index_forwarding(%v0: f32, %v1: f32, %v2: f32) -> (f32, f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloca = memref.alloca() : memref<4xf32>
  memref.store %v0, %alloca[%c0] : memref<4xf32>
  memref.store %v1, %alloca[%c1] : memref<4xf32>
  memref.store %v2, %alloca[%c2] : memref<4xf32>
  %l0 = memref.load %alloca[%c0] : memref<4xf32>
  %l1 = memref.load %alloca[%c1] : memref<4xf32>
  %l2 = memref.load %alloca[%c2] : memref<4xf32>
  return %l0, %l1, %l2 : f32, f32, f32
}

// -----

// WAW dead store elimination for indexed memrefs.
// CHECK-LABEL: func.func @indexed_dead_store
// CHECK-SAME:    (%[[V1:.*]]: f32, %[[V2:.*]]: f32)
// CHECK-NOT:     memref.store
// CHECK-NOT:     memref.alloca
// CHECK:         return
func.func @indexed_dead_store(%v1: f32, %v2: f32) {
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<4xf32>
  memref.store %v1, %alloca[%c0] : memref<4xf32>
  memref.store %v2, %alloca[%c0] : memref<4xf32>
  return
}
