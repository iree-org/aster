// RUN: aster-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @copy_different_values(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[COPY_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @copy_different_values(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.copy %arg0, %arg1 : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @copy_same_value(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr
// CHECK:         }
func.func @copy_same_value(%arg0: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = lsir.copy %arg0, %arg0 : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @copy_same_alloc(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?> {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<?>
// CHECK:         }
func.func @copy_same_alloc(%arg0: !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?> {
  lsir.copy %arg0, %arg0 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return %arg0 : !amdgcn.vgpr<?>
}

// CHECK-LABEL:   func.func @copy_different_alloc(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<?>, %[[ARG1:.*]]: !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?> {
// CHECK:           lsir.copy %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<?>
// CHECK:         }
func.func @copy_different_alloc(%arg0: !amdgcn.vgpr<?>, %arg1: !amdgcn.vgpr<?>) -> !amdgcn.vgpr<?> {
  lsir.copy %arg0, %arg1 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  return %arg0 : !amdgcn.vgpr<?>
}

// CHECK-LABEL:   func.func @copy_allocated(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0> {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<0>
// CHECK:         }
func.func @copy_allocated(%arg0: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0> {
  lsir.copy %arg0, %arg0 : !amdgcn.vgpr<0>, !amdgcn.vgpr<0>
  return %arg0 : !amdgcn.vgpr<0>
}

// CHECK-LABEL:   func.func @copy_different_allocated(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>, %[[ARG1:.*]]: !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0> {
// CHECK:           lsir.copy %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<0>
// CHECK:         }
func.func @copy_different_allocated(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>) -> !amdgcn.vgpr<0> {
  lsir.copy %arg0, %arg1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
  return %arg0 : !amdgcn.vgpr<0>
}

// CHECK-LABEL:   func.func @copy_different_same_alloc(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>, %[[ARG1:.*]]: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0> {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr<0>
// CHECK:         }
func.func @copy_different_same_alloc(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<0> {
  lsir.copy %arg0, %arg1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<0>
  return %arg0 : !amdgcn.vgpr<0>
}
