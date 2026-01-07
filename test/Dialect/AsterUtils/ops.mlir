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
