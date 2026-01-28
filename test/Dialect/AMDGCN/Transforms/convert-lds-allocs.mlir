// RUN: aster-opt --amdgcn-convert-lds-buffers %s | FileCheck %s

// CHECK-LABEL:   func.func @allocs_index_offset(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 384 : i32} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {__lds_allocation_size__ = 128 : index} 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[CONSTANT_0]] : i32 to index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_1]] {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant {__lds_allocation_size__ = 256 : index} 128 : i32
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[CONSTANT_2]] : i32 to index
// CHECK:           }
// CHECK:           scf.for %[[VAL_1:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_1]] {
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant {__lds_allocation_size__ = 128 : index} 128 : i32
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[CONSTANT_3]] : i32 to index
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @allocs_index_offset(%arg0: index, %arg1: index) attributes {gpu.shared_memory_size = 384 : i32} {
  %0 = amdgcn.alloc_lds 128 offset 0
  %c1 = arith.constant 1 : index
  %1 = amdgcn.get_lds_offset %0 : index
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %2 = amdgcn.alloc_lds 256 offset 128
    %3 = amdgcn.get_lds_offset %2 : index
    amdgcn.dealloc_lds %2
  }
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %2 = amdgcn.alloc_lds 128 offset 128
    %3 = amdgcn.get_lds_offset %2 : index
    amdgcn.dealloc_lds %2
  }
  return
}

// CHECK-LABEL:   func.func @allocs_i32_offset() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant {__lds_allocation_size__ = 16 : index} 0 : i32
// CHECK:           return
// CHECK:         }
func.func @allocs_i32_offset() {
  %0 = amdgcn.alloc_lds 16 alignment 64 offset 0
  %3 = amdgcn.get_lds_offset %0 : i32
  return
}

// CHECK-LABEL:   func.func @unallocated(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 16 alignment 64
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : i32
// CHECK:           return
// CHECK:         }
func.func @unallocated(%arg0: index) {
  %0 = amdgcn.alloc_lds 16 alignment 64
  %3 = amdgcn.get_lds_offset %0 : i32
  return
}

// CHECK-LABEL:   func.func @non_const_size(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds %[[ARG0]] alignment 64 offset 0
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : i32
// CHECK:           return
// CHECK:         }
func.func @non_const_size(%arg0: index) {
  %0 = amdgcn.alloc_lds %arg0 alignment 64 offset 0
  %3 = amdgcn.get_lds_offset %0 : i32
  return
}
