// RUN: aster-opt %s --aster-remove-assume-ops="remove-uniform=true remove-range=true remove-passthrough=true" | FileCheck %s --check-prefix=CHECK-ALL
// RUN: aster-opt %s --aster-remove-assume-ops="remove-uniform=true" | FileCheck %s --check-prefix=CHECK-UNIFORM
// RUN: aster-opt %s --aster-remove-assume-ops="remove-range=true" | FileCheck %s --check-prefix=CHECK-RANGE
// RUN: aster-opt %s --aster-remove-assume-ops="remove-passthrough=true" | FileCheck %s --check-prefix=CHECK-PASSTHROUGH
// RUN: aster-opt %s --aster-remove-assume-ops="tag-regex=debug.*" | FileCheck %s --check-prefix=CHECK-TAG-REGEX

// CHECK-ALL-LABEL: func.func @test_remove_all
// CHECK-ALL: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64)
// CHECK-ALL-NEXT: return %[[ARG0]], %[[ARG1]], %[[ARG2]]
func.func @test_remove_all(%arg0: index, %arg1: i32, %arg2: i64) -> (index, i32, i64) {
  %0 = aster_utils.assume_uniform %arg0 : index
  %1 = aster_utils.assume_range %arg1 min 0 max 32 : i32
  %2 = aster_utils.passthrough %arg2 : i64
  return %0, %1, %2 : index, i32, i64
}

// CHECK-UNIFORM-LABEL: func.func @test_remove_uniform_only
// CHECK-UNIFORM-NOT: assume_uniform
// CHECK-UNIFORM: assume_range
// CHECK-UNIFORM: passthrough
func.func @test_remove_uniform_only(%arg0: index, %arg1: i32) -> (index, i32) {
  %0 = aster_utils.assume_uniform %arg0 : index
  %1 = aster_utils.assume_range %arg1 min 0 max 32 : i32
  %2 = aster_utils.passthrough %arg1 : i32
  return %0, %2 : index, i32
}

// CHECK-RANGE-LABEL: func.func @test_remove_range_only
// CHECK-RANGE: assume_uniform
// CHECK-RANGE-NOT: assume_range
// CHECK-RANGE: passthrough
func.func @test_remove_range_only(%arg0: index, %arg1: i32) -> (index, i32) {
  %0 = aster_utils.assume_uniform %arg0 : index
  %1 = aster_utils.assume_range %arg1 min 0 max 32 : i32
  %2 = aster_utils.passthrough %arg1 : i32
  return %0, %2 : index, i32
}

// CHECK-PASSTHROUGH-LABEL: func.func @test_remove_passthrough_only
// CHECK-PASSTHROUGH: assume_uniform
// CHECK-PASSTHROUGH: assume_range
// CHECK-PASSTHROUGH-NOT: passthrough
func.func @test_remove_passthrough_only(%arg0: index, %arg1: i32) -> (index, i32) {
  %0 = aster_utils.assume_uniform %arg0 : index
  %1 = aster_utils.assume_range %arg1 min 0 max 32 : i32
  %2 = aster_utils.passthrough %arg1 : i32
  return %0, %2 : index, i32
}

// CHECK-TAG-REGEX-LABEL: func.func @test_tag_regex
// CHECK-TAG-REGEX: passthrough %arg0
// CHECK-TAG-REGEX-NOT: passthrough %arg1
// CHECK-TAG-REGEX: passthrough %arg2
func.func @test_tag_regex(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
  %0 = aster_utils.passthrough %arg0 : index
  %1 = aster_utils.passthrough %arg1 tag = "debug_point" : index
  %2 = aster_utils.passthrough %arg2 tag = "other" : index
  return %0, %1, %2 : index, index, index
}
