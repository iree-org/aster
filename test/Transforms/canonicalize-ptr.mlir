// RUN: aster-opt %s -aster-canonicalize-ptr --split-input-file | FileCheck %s

!ptr = !ptr.ptr<#ptr.generic_space>

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL:   func.func @fold_consecutive_ptr_add_index(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_consecutive_ptr_add_index(%arg0: !ptr, %arg1: index, %arg2: index) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, index
  %1 = ptr.ptr_add %0, %arg2 : !ptr, index
  return %1 : !ptr
}

// CHECK-LABEL:   func.func @fold_consecutive_ptr_add_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_consecutive_ptr_add_i32(%arg0: !ptr, %arg1: i32, %arg2: i32) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, i32
  %1 = ptr.ptr_add %0, %arg2 : !ptr, i32
  return %1 : !ptr
}

// CHECK-LABEL:   func.func @fold_consecutive_ptr_add_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_0]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_consecutive_ptr_add_i64(%arg0: !ptr, %arg1: i64, %arg2: i64) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, i64
  %1 = ptr.ptr_add %0, %arg2 : !ptr, i64
  return %1 : !ptr
}

// CHECK-LABEL:   func.func @no_fold_different_types(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ARG1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[ARG2]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @no_fold_different_types(%arg0: !ptr, %arg1: i32, %arg2: i64) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, i32
  %1 = ptr.ptr_add %0, %arg2 : !ptr, i64
  return %1 : !ptr
}

// CHECK-LABEL:   func.func @fold_with_constants_index(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 15 : index
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[CONSTANT_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_with_constants_index(%arg0: !ptr) -> !ptr {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %0 = ptr.ptr_add %arg0, %c5 : !ptr, index
  %1 = ptr.ptr_add %0, %c10 : !ptr, index
  return %1 : !ptr
}

// CHECK-LABEL:   func.func @fold_zero(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           return %[[ARG0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_zero(%arg0: !ptr, %arg1: index) -> !ptr {
  %c0 = arith.constant 0 : index
  %0 = ptr.ptr_add %arg0, %c0 : !ptr, index
  return %0 : !ptr
}

// CHECK-LABEL:   func.func @fold_vector_ptr_add(
// CHECK-SAME:      %[[ARG0:.*]]: vector<4x!ptr.ptr<#ptr.generic_space>>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> vector<4x!ptr.ptr<#ptr.generic_space>> {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_0]] : vector<4x!ptr.ptr<#ptr.generic_space>>, i32
// CHECK:           return %[[PTR_ADD_0]] : vector<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:         }
func.func @fold_vector_ptr_add(%arg0: vector<4x!ptr>, %arg1: i32, %arg2: i32) -> vector<4x!ptr> {
  %0 = ptr.ptr_add %arg0, %arg1 : vector<4x!ptr>, i32
  %1 = ptr.ptr_add %0, %arg2 : vector<4x!ptr>, i32
  return %1 : vector<4x!ptr>
}

// CHECK-LABEL:   func.func @fold_multiple_uses_base(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> (!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>) {
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ARG1]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]], %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_multiple_uses_base(%arg0: !ptr, %arg1: index, %arg2: index) -> (!ptr, !ptr) {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, index
  %1 = ptr.ptr_add %0, %arg2 : !ptr, index
  return %0, %1 : !ptr, !ptr
}

// CHECK-LABEL:   func.func @fold_nested_chain(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i64
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[ARG3]] : i64
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[ARG4]] : i64
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[ADDI_2]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_nested_chain(%arg0: !ptr, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, i64
  %1 = ptr.ptr_add %0, %arg2 : !ptr, i64
  %2 = ptr.ptr_add %1, %arg3 : !ptr, i64
  %3 = ptr.ptr_add %2, %arg4 : !ptr, i64
  return %3 : !ptr
}

// -----
!ptr = !ptr.ptr<#ptr.generic_space>

// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
// CHECK-LABEL:   func.func @fold_three_consecutive_ptr_add(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[ARG3]], %[[ARG1]], %[[ARG2]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @fold_three_consecutive_ptr_add(%arg0: !ptr, %arg1: index, %arg2: index, %arg3: index) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, index
  %1 = ptr.ptr_add %0, %arg2 : !ptr, index
  %2 = ptr.ptr_add %1, %arg3 : !ptr, index
  return %2 : !ptr
}

// -----
!ptr = !ptr.ptr<#ptr.generic_space>

// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:   func.func @type_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[ARG1:.*]]: index) -> !ptr.ptr<#ptr.generic_space> {
// CHECK:           %[[APPLY_0:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[ARG1]]]
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[ARG0]], %[[APPLY_0]] : !ptr.ptr<#ptr.generic_space>, index
// CHECK:           return %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:         }
func.func @type_offset(%arg0: !ptr, %arg1: index) -> !ptr {
  %0 = ptr.ptr_add %arg0, %arg1 : !ptr, index
  %1 = ptr.type_offset i32 : index
  %2 = ptr.ptr_add %0, %1 : !ptr, index
  return %2 : !ptr
}
