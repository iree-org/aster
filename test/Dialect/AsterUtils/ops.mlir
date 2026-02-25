// RUN: aster-opt %s --verify-roundtrip

// Test to_any roundtrip
func.func @test_to_any_i32(%arg: i32) -> !aster_utils.any {
  %result = aster_utils.to_any %arg : i32
  return %result : !aster_utils.any
}

func.func @test_to_any_f32(%arg: f32) -> !aster_utils.any {
  %result = aster_utils.to_any %arg : f32
  return %result : !aster_utils.any
}

// Test from_any roundtrip
func.func @test_from_any_i32(%arg: !aster_utils.any) -> i32 {
  %result = aster_utils.from_any %arg : i32
  return %result : i32
}

func.func @test_from_any_f32(%arg: !aster_utils.any) -> f32 {
  %result = aster_utils.from_any %arg : f32
  return %result : f32
}

// Test chained to_any and from_any roundtrip
func.func @test_to_from_any_chain(%arg: i32) -> i64 {
  %any1 = aster_utils.to_any %arg : i32
  %val1 = aster_utils.from_any %any1 : i32
  %any2 = aster_utils.to_any %val1 : i32
  %val2 = aster_utils.from_any %any2 : i64
  return %val2 : i64
}

//===----------------------------------------------------------------------===//
// AssumeRangeOp roundtrip tests
//===----------------------------------------------------------------------===//

// Static min and max (original syntax)
func.func @test_assume_range_static_both(%x: i64) -> i64 {
  %0 = aster_utils.assume_range %x min 0 max 1024 : i64
  return %0 : i64
}

// Static min only
func.func @test_assume_range_static_min_only(%x: i32) -> i32 {
  %0 = aster_utils.assume_range %x min 0 : i32
  return %0 : i32
}

// Static max only
func.func @test_assume_range_static_max_only(%x: i32) -> i32 {
  %0 = aster_utils.assume_range %x max 256 : i32
  return %0 : i32
}

// No bounds (identity)
func.func @test_assume_range_no_bounds(%x: i64) -> i64 {
  %0 = aster_utils.assume_range %x : i64
  return %0 : i64
}

// Dynamic min, static max
func.func @test_assume_range_dynamic_min_static_max(%x: i64, %lo: i64) -> i64 {
  %0 = aster_utils.assume_range %x min %lo max 1024 : i64
  return %0 : i64
}

// Static min, dynamic max
func.func @test_assume_range_static_min_dynamic_max(%x: i64, %hi: i64) -> i64 {
  %0 = aster_utils.assume_range %x min 0 max %hi : i64
  return %0 : i64
}

// Both dynamic
func.func @test_assume_range_dynamic_both(%x: i64, %lo: i64, %hi: i64) -> i64 {
  %0 = aster_utils.assume_range %x min %lo max %hi : i64
  return %0 : i64
}

// Dynamic min only
func.func @test_assume_range_dynamic_min_only(%x: i32, %lo: i32) -> i32 {
  %0 = aster_utils.assume_range %x min %lo : i32
  return %0 : i32
}

// Dynamic max only
func.func @test_assume_range_dynamic_max_only(%x: i32, %hi: i32) -> i32 {
  %0 = aster_utils.assume_range %x max %hi : i32
  return %0 : i32
}

// Index type with dynamic bounds
func.func @test_assume_range_index_dynamic(%x: index, %lo: index, %hi: index) -> index {
  %0 = aster_utils.assume_range %x min %lo max %hi : index
  return %0 : index
}
