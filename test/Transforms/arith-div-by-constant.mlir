// RUN: aster-opt %s --aster-optimize-arith | FileCheck %s

// CHECK-LABEL: func.func @sdiv_by_3
// CHECK-SAME:    %[[N:.*]]: i32
// CHECK-DAG:     %[[M:.*]] = arith.constant 1431655766 : i32
// CHECK-DAG:     %[[C31:.*]] = arith.constant 31 : i32
// CHECK:         %{{.*}}, %[[HI:.*]] = arith.mulsi_extended %[[N]], %[[M]] : i32
// CHECK:         %[[SIGN:.*]] = arith.shrui %[[HI]], %[[C31]] : i32
// CHECK:         %[[R:.*]] = arith.addi %[[HI]], %[[SIGN]] : i32
// CHECK:         return %[[R]]
func.func @sdiv_by_3(%n: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %0 = arith.divsi %n, %c3 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @sdiv_by_5
// CHECK:         arith.mulsi_extended
// CHECK:         arith.shrsi
// CHECK:         arith.shrui
// CHECK:         arith.addi
func.func @sdiv_by_5(%n: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %0 = arith.divsi %n, %c5 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @sdiv_by_7
// CHECK:         arith.mulsi_extended
// CHECK:         arith.addi
// CHECK:         arith.shrsi
// CHECK:         arith.shrui
// CHECK:         arith.addi
func.func @sdiv_by_7(%n: i32) -> i32 {
  %c7 = arith.constant 7 : i32
  %0 = arith.divsi %n, %c7 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @sdiv_by_6
// CHECK:         arith.mulsi_extended
// CHECK-NOT:     arith.divsi
func.func @sdiv_by_6(%n: i32) -> i32 {
  %c6 = arith.constant 6 : i32
  %0 = arith.divsi %n, %c6 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @sdiv_by_4_uses_shift
// CHECK:         arith.shrsi
// CHECK-NOT:     arith.mulsi_extended
func.func @sdiv_by_4_uses_shift(%n: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %0 = arith.divsi %n, %c4 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @srem_by_3
// CHECK-SAME:    %[[N:.*]]: i32
// CHECK:         arith.mulsi_extended
// CHECK:         arith.muli
// CHECK:         arith.subi %[[N]]
// CHECK-NOT:     arith.remsi
func.func @srem_by_3(%n: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %0 = arith.remsi %n, %c3 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @srem_by_4_uses_and
// CHECK:         arith.andi
// CHECK-NOT:     arith.mulsi_extended
func.func @srem_by_4_uses_and(%n: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %0 = arith.remsi %n, %c4 : i32
  return %0 : i32
}
