// RUN: aster-opt %s | aster-opt | FileCheck %s

// ---- Flat layouts ----

// CHECK-LABEL: func.func @test_linearize_rank1
func.func @test_linearize_rank1(%c: index) -> index {
  // CHECK: layout.linearize %{{.*}}, <[64] : [16]>
  %off = layout.linearize %c, #layout.strided_layout<[64] : [16]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_rank2_rowmajor
func.func @test_linearize_rank2_rowmajor(%c: index) -> index {
  // Row-major 4x8: first mode (4 rows) has stride 8, second (8 cols) stride 1.
  // CHECK: layout.linearize %{{.*}}, <[4, 8] : [8, 1]>
  %off = layout.linearize %c, #layout.strided_layout<[4, 8] : [8, 1]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_rank2_colmajor
func.func @test_linearize_rank2_colmajor(%c: index) -> index {
  // Column-major 4x8: first mode (4 rows) has stride 1, second (8 cols) stride 4.
  // CHECK: layout.linearize %{{.*}}, <[4, 8] : [1, 4]>
  %off = layout.linearize %c, #layout.strided_layout<[4, 8] : [1, 4]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_broadcast
func.func @test_linearize_broadcast(%c: index) -> index {
  // CHECK: layout.linearize %{{.*}}, <[4, 16] : [16, 0]>
  %off = layout.linearize %c, #layout.strided_layout<[4, 16] : [16, 0]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_rank3
func.func @test_linearize_rank3(%c: index) -> index {
  // CHECK: layout.linearize %{{.*}}, <[2, 2, 2] : [1, 2, 4]>
  %off = layout.linearize %c, #layout.strided_layout<[2, 2, 2] : [1, 2, 4]>
  return %off : index
}

// ---- Nested layouts ----

// CHECK-LABEL: func.func @test_linearize_nested_rank2
func.func @test_linearize_nested_rank2(%c: index) -> index {
  // CHECK: layout.linearize %{{.*}}, <[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>
  %off = layout.linearize %c, #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_mixed_nesting
func.func @test_linearize_mixed_nesting(%c: index) -> index {
  // One mode flat, one mode nested.
  // CHECK: layout.linearize %{{.*}}, <[4, (2, 4)] : [1, (4, 8)]>
  %off = layout.linearize %c, #layout.strided_layout<[4, (2, 4)] : [1, (4, 8)]>
  return %off : index
}

// CHECK-LABEL: func.func @test_linearize_deeply_nested
func.func @test_linearize_deeply_nested(%c: index) -> index {
  // CHECK: layout.linearize %{{.*}}, <[(2, (2, 2)), 8] : [(1, (4, 16)), 64]>
  %off = layout.linearize %c, #layout.strided_layout<[(2, (2, 2)), 8] : [(1, (4, 16)), 64]>
  return %off : index
}

// ---- Swizzle ----

// CHECK-LABEL: func.func @test_swizzle
func.func @test_swizzle(%off: index) -> index {
  // CHECK: layout.swizzle %{{.*}}, bits = 2, base = 3, shift = 4
  %r = layout.swizzle %off, bits = 2, base = 3, shift = 4
  return %r : index
}

// CHECK-LABEL: func.func @test_swizzle_different_params
func.func @test_swizzle_different_params(%off: index) -> index {
  // CHECK: layout.swizzle %{{.*}}, bits = 3, base = 1, shift = 5
  %r = layout.swizzle %off, bits = 3, base = 1, shift = 5
  return %r : index
}
