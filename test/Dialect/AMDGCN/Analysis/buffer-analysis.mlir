// RUN: aster-opt %s --test-buffer-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: func @simple_buffer_lifetime
func.func @simple_buffer_lifetime(%arg0: index) {
  %0 = amdgcn.alloc_lds 32
// CHECK: %[[ALLOC:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
  amdgcn.dealloc_lds %0
  return
}

// CHECK-LABEL: func @buffer_with_ptr_usage
func.func @buffer_with_ptr_usage(%arg0: index) -> i32 {
  %0 = amdgcn.alloc_lds %arg0
// CHECK: %[[ALLOC:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
  %1 = amdgcn.get_lds_offset %0 : i32
  amdgcn.dealloc_lds %0
  return %1 : i32
}

// CHECK-LABEL: func @multiple_buffers
func.func @multiple_buffers(%arg0: index) -> (i32, i32) {
  %c64 = arith.constant 64 : index
  %0 = amdgcn.alloc_lds %arg0
// CHECK: %[[ALLOC0:.*]] = amdgcn.alloc_lds
  %1 = amdgcn.alloc_lds %c64
// CHECK: %[[ALLOC1:.*]] = amdgcn.alloc_lds
  %2 = amdgcn.get_lds_offset %0 : i32
  %3 = amdgcn.get_lds_offset %1 : i32
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC0]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC0]], live}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %0
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC1]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], dead}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %1
  return %2, %3 : i32, i32
}

// CHECK-LABEL: func @sequential_buffers
func.func @sequential_buffers(%arg0: index, %arg1: index) {
  %0 = amdgcn.alloc_lds %arg0
// CHECK: %[[ALLOC0:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC0]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC0]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC0]], dead}]
  amdgcn.dealloc_lds %0
  %1 = amdgcn.alloc_lds %arg1
// CHECK: %[[ALLOC1:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC1]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC1]], dead}
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %1
  return
}

// CHECK-LABEL: func @invalid_liveness
func.func @invalid_liveness(%arg0: index) {
  %0 = amdgcn.alloc_lds %arg0
// CHECK: %[[ALLOC0:.*]] = amdgcn.alloc_lds
  %1 = amdgcn.alloc_lds %arg0
// CHECK: %[[ALLOC1:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC0]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC0]], live}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %0
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC1]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], live}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], dead}
// CHECK-DAG: {%[[ALLOC1]], dead}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %1
  return
}

// CHECK-LABEL: func @buffer_in_loop
func.func @buffer_in_loop(%arg0: index, %arg1: index, %arg2: index) {
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %arg0 to %arg1 step %c1 {
    %0 = amdgcn.alloc_lds %arg2
// CHECK: %[[ALLOC:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
    %1 = amdgcn.get_lds_offset %0 : i32
    amdgcn.dealloc_lds %0
  }
  return
}

// CHECK-LABEL: func @buffer_in_loop_1
func.func @buffer_in_loop_1(%arg0: index, %arg1: index, %arg2: index) {
  %c1 = arith.constant 1 : index
// CHECK: %[[ALLOC0:.*]] = amdgcn.alloc_lds
  %0 = amdgcn.alloc_lds %arg2
// CHECK: %[[FOR:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[ALLOC0]]) -> (!amdgcn.lds_buffer) {...}
// CHECK-NEXT: BEFORE: [{%[[ALLOC0]], live}]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], live}
// CHECK-DAG: {%[[ITER]], top}
// CHECK-SAME: ]
  %1 = scf.for %arg3 = %arg0 to %arg1 step %c1 iter_args(%arg4 = %0) -> (!amdgcn.lds_buffer) {
    %2 = amdgcn.alloc_lds %arg2
    %3 = amdgcn.get_lds_offset %2 : i32
    scf.yield %2 : !amdgcn.lds_buffer
  }
// CHECK: Op: amdgcn.dealloc_lds %[[FOR]]
// CHECK-NEXT: BEFORE: [
// CHECK-DAG: {%[[ALLOC0]], live}
// CHECK-DAG: {%[[FOR]], top}
// CHECK-SAME: ]
// CHECK-NEXT: AFTER: [
// CHECK-DAG: {%[[ALLOC0]], live}
// CHECK-DAG: {%[[FOR]], top}
// CHECK-SAME: ]
  amdgcn.dealloc_lds %1
  return
}

// CHECK-LABEL: func @buffer_in_if
func.func @buffer_in_if(%arg0: i1) {
  scf.if %arg0 {
    %0 = amdgcn.alloc_lds 32
// CHECK: %[[ALLOC:.*]] = amdgcn.alloc_lds
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
    %1 = amdgcn.get_lds_offset %0 : i32
    amdgcn.dealloc_lds %0
  }
  return
}

// CHECK-LABEL: func @buffer_in_if_1
func.func @buffer_in_if_1(%arg0: i1) {
  %0 = amdgcn.alloc_lds 32
// CHECK: %[[ALLOC:.*]] = amdgcn.alloc_lds
  scf.if %arg0 {
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
    %1 = amdgcn.get_lds_offset %0 : i32
    amdgcn.dealloc_lds %0
  } else {
// CHECK: Op: amdgcn.dealloc_lds %[[ALLOC]]
// CHECK-NEXT: BEFORE: [{%[[ALLOC]], live}]
// CHECK-NEXT: AFTER: [{%[[ALLOC]], dead}]
    amdgcn.dealloc_lds %0
  }
// CHECK: AFTER: [{%[[ALLOC]], dead}]
  return
}

// CHECK-LABEL: func @buffer_in_if_dead
func.func @buffer_in_if_dead(%arg0: i1) {
// CHECK: %[[RES:.*]] = scf.if %{{.*}}
// CHECK-NEXT: BEFORE: []
// CHECK-NEXT: AFTER: [{%[[RES]], dead}]
  scf.if %arg0 -> !amdgcn.lds_buffer {
    %0 = amdgcn.alloc_lds 32
    scf.yield %0 : !amdgcn.lds_buffer
  } else {
    %0 = amdgcn.alloc_lds 32
    scf.yield %0 : !amdgcn.lds_buffer
  }
  return
}
