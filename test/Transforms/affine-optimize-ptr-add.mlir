// RUN: aster-opt %s --aster-affine-optimize-ptr-add="assume-positive=true" --canonicalize --cse --split-input-file | FileCheck %s

#map = affine_map<()[s0, s1] -> (s0 * 8 + (s1 floordiv 32) * 1048576 - (s0 floordiv 64) * 512 + ((s0 floordiv 64) floordiv 2) * 524288 + ((s0 mod 64) floordiv 8) * 8128)>
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 64) * 512 + ((s0 floordiv 64) floordiv 2) * 524288 + ((s0 mod 64) floordiv 8) * 8128)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> ((s0 floordiv 32) * 1048576)>
// CHECK-LABEL:   func.func @test_dynamic_uniform(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG2]] : index
// CHECK:           %[[ASSUME_RANGE_0:.*]] = aster_utils.assume_range %[[ASSUME_UNIFORM_0]] min 0 : index
// CHECK:           %[[ASSUME_RANGE_1:.*]] = aster_utils.assume_range %[[ARG1]] min 0 max 255 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ASSUME_RANGE_1]]]
// CHECK:           %[[APPLY_1:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[ASSUME_RANGE_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_1]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_dynamic_uniform(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index, %arg2: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg2 : index
  %1 = aster_utils.assume_range %0 min 0 : index
  %2 = aster_utils.assume_range %arg1 min 0 max 255 : index
  %3 = affine.apply #map()[%2, %1]
  %4 = ptr.ptr_add %arg0, %3 : !ptr.ptr<#ptr.generic_space>, index
  return %4 : !ptr.ptr<#ptr.generic_space>
}

// -----

#map = affine_map<()[s0, s1, s2] -> (((s0 + s2 + 16) * s1) * 4 + 32)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) * 4)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) * 4 + s1 * 64)>
// CHECK-LABEL:   func.func @test_dynamic_uniform_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : index
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG2]] : index
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG3]] : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[ASSUME_UNIFORM_1]], %[[ARG1]]]
// CHECK:           %[[APPLY_1:.*]] = affine.apply #[[$ATTR_3]](){{\[}}%[[ASSUME_UNIFORM_0]], %[[ASSUME_UNIFORM_1]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_1]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[PTR_ADD_1]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_dynamic_uniform_const(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index, %arg2: index, %arg3: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg2 : index
  %1 = aster_utils.assume_uniform %arg3 : index
  %2 = affine.apply #map()[%0, %1, %arg1]
  %3 = ptr.ptr_add %arg0, %2 : !ptr.ptr<#ptr.generic_space>, index
  return %3 : !ptr.ptr<#ptr.generic_space>
}

// -----

#map = affine_map<()[s0] -> (s0 * 32 + 64)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @test_dynamic_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_4]](){{\[}}%[[ARG1]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add inbounds %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add inbounds %[[PTR_ADD_0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_dynamic_const(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index, %arg2: index, %arg3: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = affine.apply #map()[%arg1]
  %1 = ptr.ptr_add inbounds %arg0, %0 : !ptr.ptr<#ptr.generic_space>, index
  return %1 : !ptr.ptr<#ptr.generic_space>
}

// -----

#map = affine_map<()[s0, s1] -> ((s1 + 32) * (s0 + 16))>
// CHECK: #[[$ATTR_5:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 16 + s0 * s1)>
// CHECK-LABEL:   func.func @test_uniform_const(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 512 : index
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : index
// CHECK:           %[[ASSUME_UNIFORM_1:.*]] = aster_utils.assume_uniform %[[ARG2]] : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_5]](){{\[}}%[[ASSUME_UNIFORM_0]], %[[ASSUME_UNIFORM_1]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_uniform_const(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index, %arg2: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : index
  %1 = aster_utils.assume_uniform %arg2 : index
  %2 = affine.apply #map()[%0, %1]
  %3 = ptr.ptr_add %arg0, %2 : !ptr.ptr<#ptr.generic_space>, index
  return %3 : !ptr.ptr<#ptr.generic_space>
}

// -----

#map = affine_map<()[s0] -> (s0 * 32)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL:   func.func @test_no_change(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ASSUME_UNIFORM_0:.*]] = aster_utils.assume_uniform %[[ARG1]] : index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_6]](){{\[}}%[[ASSUME_UNIFORM_0]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @test_no_change(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: index) -> !ptr.ptr<#ptr.generic_space> {
  %0 = aster_utils.assume_uniform %arg1 : index
  %1 = affine.apply #map()[%0]
  %2 = ptr.ptr_add %arg0, %1 : !ptr.ptr<#ptr.generic_space>, index
  return %2 : !ptr.ptr<#ptr.generic_space>
}
