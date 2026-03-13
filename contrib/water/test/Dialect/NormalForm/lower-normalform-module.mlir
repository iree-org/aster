// RUN: water-opt %s -lower-water_normalform-module --mlir-print-local-scope --split-input-file | FileCheck %s

//-----------------------------------------------------------------------------
// Test lowering of water_normalform.module to builtin.module.
//-----------------------------------------------------------------------------

// Test that a top-level water_normalform.module is inlined into the root module.
// CHECK: module {
// CHECK-NOT: water_normalform.module
// CHECK-NOT: module {
// CHECK:   func.func @inlined_into_root()
// CHECK: }
water_normalform.module [] {
  func.func @inlined_into_root() {
    return
  }
}

// -----

// Test that a named water_normalform.module is inlined into the root module.
// CHECK: module {
// CHECK-NOT: water_normalform.module
// CHECK-NOT: module {
// CHECK:   func.func @from_named_module()
// CHECK: }
water_normalform.module @named [] {
  func.func @from_named_module() {
    return
  }
}

// -----

// Test that multiple operations are preserved when inlining.
// CHECK: module {
// CHECK:   func.func @first()
// CHECK:   func.func @second()
// CHECK:   func.func @third()
// CHECK: }
water_normalform.module [] {
  func.func @first() {
    return
  }
  func.func @second() {
    return
  }
  func.func @third() {
    return
  }
}
