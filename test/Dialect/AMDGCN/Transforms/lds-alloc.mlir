// RUN: aster-opt --pass-pipeline="builtin.module(func.func(amdgcn-lds-alloc))" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @sequential_buffers(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 64 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 64 offset 0
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : i32
// CHECK:           amdgcn.dealloc_lds %[[ALLOC_LDS_0]]
// CHECK:           %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 32 offset 0
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : i32
// CHECK:           amdgcn.dealloc_lds %[[ALLOC_LDS_1]]
// CHECK:           return
// CHECK:         }
func.func @sequential_buffers(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 64
  %1 = amdgcn.get_lds_offset %0 : i32
  amdgcn.dealloc_lds %0
  %2 = amdgcn.alloc_lds 32
  %3 = amdgcn.get_lds_offset %2 : i32
  amdgcn.dealloc_lds %2
  return
}

// CHECK-LABEL:   func.func @basic_overlap(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 128 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 64 offset 0
// CHECK:           %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 64 offset 64
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : index
// CHECK:           amdgcn.dealloc_lds %[[ALLOC_LDS_1]]
// CHECK:           %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 32 offset 64
// CHECK:           %[[GET_LDS_OFFSET_2:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_2]] : i32
// CHECK:           amdgcn.dealloc_lds %[[ALLOC_LDS_2]]
// CHECK:           return
// CHECK:         }
func.func @basic_overlap(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 64
  %1 = amdgcn.alloc_lds 64
  %2 = amdgcn.get_lds_offset %0 : index
  %3 = amdgcn.get_lds_offset %1 : index
  amdgcn.dealloc_lds %1
  %4 = amdgcn.alloc_lds 32
  %5 = amdgcn.get_lds_offset %4 : i32
  amdgcn.dealloc_lds %4
  return
}

// CHECK-LABEL:   func.func @buffer_in_loop(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 128 : i32} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 128 offset 0
// CHECK:             %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:             amdgcn.dealloc_lds %[[ALLOC_LDS_0]]
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @buffer_in_loop(%arg0: index, %arg1: index) {
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %0 = amdgcn.alloc_lds 128
    %1 = amdgcn.get_lds_offset %0 : index
    amdgcn.dealloc_lds %0
  }
  return
}

// CHECK-LABEL:   func.func @buffer_in_loop_1(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 384 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 128 offset 0
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 256 offset 128
// CHECK:             %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : index
// CHECK:             amdgcn.dealloc_lds %[[ALLOC_LDS_1]]
// CHECK:           }
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:           return
// CHECK:         }
func.func @buffer_in_loop_1(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 128
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %2 = amdgcn.alloc_lds 256
    %3 = amdgcn.get_lds_offset %2 : index
    amdgcn.dealloc_lds %2
  }
  %1 = amdgcn.get_lds_offset %0 : index
  return
}

// CHECK-LABEL:   func.func @buffer_in_loop_2(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 384 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 128 offset 0
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 256 offset 128
// CHECK:             %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : index
// CHECK:             amdgcn.dealloc_lds %[[ALLOC_LDS_1]]
// CHECK:           }
// CHECK:           scf.for %[[VAL_1:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK:             %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 128 offset 128
// CHECK:             %[[GET_LDS_OFFSET_2:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_2]] : index
// CHECK:             amdgcn.dealloc_lds %[[ALLOC_LDS_2]]
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @buffer_in_loop_2(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 128
  %c1 = arith.constant 1 : index
  %1 = amdgcn.get_lds_offset %0 : index
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %2 = amdgcn.alloc_lds 256
    %3 = amdgcn.get_lds_offset %2 : index
    amdgcn.dealloc_lds %2
  }
  scf.for %arg2 = %arg0 to %arg1 step %c1 {
    %2 = amdgcn.alloc_lds 128
    %3 = amdgcn.get_lds_offset %2 : index
    amdgcn.dealloc_lds %2
  }
  return
}

// CHECK-LABEL:   func.func @buffer_in_loop_3(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 640 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 128 offset 0
// CHECK:           %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 256 offset 128
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : i32
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : i32
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[GET_LDS_OFFSET_0]], %[[VAL_2:.*]] = %[[GET_LDS_OFFSET_1]]) -> (i32, i32) {
// CHECK:             scf.yield %[[VAL_2]], %[[VAL_1]] : i32, i32
// CHECK:           }
// CHECK:           amdgcn.dealloc_lds %[[ALLOC_LDS_1]]
// CHECK:           %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 512 offset 128
// CHECK:           %[[GET_LDS_OFFSET_2:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_2]] : index
// CHECK:           return
// CHECK:         }
func.func @buffer_in_loop_3(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 128
  %1 = amdgcn.alloc_lds 256
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %2 = amdgcn.get_lds_offset %0 : i32
  %3 = amdgcn.get_lds_offset %1 : i32
  %4:2 = scf.for %arg2 = %arg0 to %arg1 step %c1 iter_args(%arg3 = %2, %arg4 = %3) -> (i32, i32) {
    scf.yield %arg4, %arg3 : i32, i32
  }
  amdgcn.dealloc_lds %1
  %5 = amdgcn.alloc_lds 512
  %6 = amdgcn.get_lds_offset %5 : index
  return
}

// CHECK-LABEL:   func.func @alignments(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 80 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 16 alignment 64 offset 0
// CHECK:           %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 16 alignment 64 offset 64
// CHECK:           %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 8 alignment 8 offset 16
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : index
// CHECK:           %[[GET_LDS_OFFSET_2:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_2]] : index
// CHECK:           return
// CHECK:         }
func.func @alignments(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 16 alignment 64
  %1 = amdgcn.alloc_lds 16 alignment 64
  %2 = amdgcn.alloc_lds 8 alignment 8
  %3 = amdgcn.get_lds_offset %0 : index
  %4 = amdgcn.get_lds_offset %1 : index
  %5 = amdgcn.get_lds_offset %2 : index
  return
}

// CHECK-LABEL:   func.func @allocated(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index) attributes {gpu.shared_memory_size = 40 : i32} {
// CHECK:           %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds 16 alignment 8 offset 16
// CHECK:           %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 16 alignment 64 offset 0
// CHECK:           %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 8 alignment 8 offset 32
// CHECK:           %[[GET_LDS_OFFSET_0:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_0]] : index
// CHECK:           %[[GET_LDS_OFFSET_1:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_1]] : index
// CHECK:           %[[GET_LDS_OFFSET_2:.*]] = amdgcn.get_lds_offset %[[ALLOC_LDS_2]] : index
// CHECK:           return
// CHECK:         }
func.func @allocated(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds 16 alignment 8 offset 16
  %1 = amdgcn.alloc_lds 16 alignment 64
  %2 = amdgcn.alloc_lds 8 alignment 8
  %3 = amdgcn.get_lds_offset %0 : index
  %4 = amdgcn.get_lds_offset %1 : index
  %5 = amdgcn.get_lds_offset %2 : index
  return
}

// -----

func.func @max_memory() {
  //  expected-error@+1 {{failed to allocate LDS buffer of size 66000 with alignment 16}}
  %0 = amdgcn.alloc_lds 66000
  return
}

// -----

func.func @impossible_to_satisfy_alloc() {
  %0 = amdgcn.alloc_lds 16000 offset 24000
  //  expected-error@+1 {{failed to allocate LDS buffer of size 32000 with alignment 16}}
  %1 = amdgcn.alloc_lds 32000
  return
}
