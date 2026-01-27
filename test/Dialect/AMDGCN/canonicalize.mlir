// RUN: aster-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @merge_waits(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.read_token<flat>, %[[ARG1:.*]]: !amdgcn.read_token<shared>, %[[ARG2:.*]]: !amdgcn.write_token<flat>) {
// CHECK:           amdgcn.wait vm_cnt 0 lgkm_cnt 1 deps %[[ARG0]], %[[ARG1]], %[[ARG2]] : !amdgcn.read_token<flat>, !amdgcn.read_token<shared>, !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @merge_waits(
    %rt1: !amdgcn.read_token<flat>,
    %rt2: !amdgcn.read_token<shared>,
    %wt1: !amdgcn.write_token<flat>) {
  amdgcn.wait deps %rt1 : !amdgcn.read_token<flat>
  amdgcn.wait deps %rt1, %rt2 : !amdgcn.read_token<flat>, !amdgcn.read_token<shared>
  amdgcn.wait deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  amdgcn.wait vm_cnt 0 lgkm_cnt 1 deps %rt1, %wt1 : !amdgcn.read_token<flat>, !amdgcn.write_token<flat>
  amdgcn.wait vm_cnt 2
  return
}

// CHECK-LABEL:   func.func @remove_duplicate_waits(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.read_token<flat>) {
// CHECK:           amdgcn.wait deps %[[ARG0]] : !amdgcn.read_token<flat>
// CHECK:           return
// CHECK:         }
func.func @remove_duplicate_waits(%rt1: !amdgcn.read_token<flat>) {
  amdgcn.wait deps %rt1, %rt1, %rt1 : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @erase_noop_wait() {
// CHECK:           return
// CHECK:         }
func.func @erase_noop_wait() {
  amdgcn.wait
  return
}

// CHECK-LABEL:   func.func @lds_buffer_folding(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> (!amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer) {
// CHECK-DAG:       %[[ALLOC_LDS_0:.*]] = amdgcn.alloc_lds %[[ARG0]]
// CHECK-DAG:       %[[ALLOC_LDS_1:.*]] = amdgcn.alloc_lds 64
// CHECK-DAG:       %[[ALLOC_LDS_2:.*]] = amdgcn.alloc_lds 32
// CHECK:           return %[[ALLOC_LDS_0]], %[[ALLOC_LDS_1]], %[[ALLOC_LDS_2]] : !amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer
// CHECK:         }
func.func @lds_buffer_folding(%arg0: index) -> (!amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer) {
  %c64 = arith.constant 64 : index
  %0 = amdgcn.alloc_lds %arg0
  %1 = amdgcn.alloc_lds %c64
  %2 = amdgcn.alloc_lds 32
  return %0, %1, %2 : !amdgcn.lds_buffer, !amdgcn.lds_buffer, !amdgcn.lds_buffer
}
