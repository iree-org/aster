// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Both static and dynamic min is invalid.
func.func @both_static_and_dynamic_min(%x: i64, %lo: i64) -> i64 {
  // expected-error @below {{cannot have both static and dynamic min}}
  %0 = "aster_utils.assume_range"(%x, %lo) <{static_min = 0 : index, operandSegmentSizes = array<i32: 1, 1, 0>}> : (i64, i64) -> i64
  return %0 : i64
}

// -----

// Both static and dynamic max is invalid.
func.func @both_static_and_dynamic_max(%x: i64, %hi: i64) -> i64 {
  // expected-error @below {{cannot have both static and dynamic max}}
  %0 = "aster_utils.assume_range"(%x, %hi) <{static_max = 1024 : index, operandSegmentSizes = array<i32: 1, 0, 1>}> : (i64, i64) -> i64
  return %0 : i64
}

// -----

// Both static and dynamic on both bounds.
func.func @both_static_and_dynamic_min_and_max(%x: i64, %lo: i64, %hi: i64) -> i64 {
  // expected-error @below {{cannot have both static and dynamic min}}
  %0 = "aster_utils.assume_range"(%x, %lo, %hi) <{static_min = 0 : index, static_max = 1024 : index, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i64, i64, i64) -> i64
  return %0 : i64
}
