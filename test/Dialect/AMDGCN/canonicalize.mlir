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
