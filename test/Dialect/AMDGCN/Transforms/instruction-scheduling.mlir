// RUN: aster-opt --amdgcn-instruction-scheduling="test-only=true" --allow-unregistered-dialect --mlir-disable-threading %s | FileCheck %s

// CHECK-LABEL: func.func @test_scheduling()
// CHECK-NOT: scf.for
//       CHECK:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} 3 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 1 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 2 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 2 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 3 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 3 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 3 : i32} 3 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 4 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 4 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 4 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} 0 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 5 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 6 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 6 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 6 : i32} 3 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 7 : i32} 1 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 8 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 9 : i32} 3 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 10 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 12 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 12 : i32} 3 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 14 : i32} 2 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 15 : i32} 3 : i32
//  CHECK-NEXT:   return
func.func @test_scheduling() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  // Loop with static bounds equivalent to dimensions [1, 2, 3] -> 6 iterations
  scf.for %i = %c0 to %c6 step %c1 {
    %c0_i32 = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 0 : i32
    %c1_i32 = arith.constant {sched.delay = 2 : i32, sched.rate = 1 : i32} 1 : i32
    %c2_i32 = arith.constant {sched.delay = 4 : i32, sched.rate = 2 : i32} 2 : i32
    %c3_i32 = arith.constant {sched.delay = 0 : i32, sched.rate = 3 : i32} 3 : i32
  } {sched.dims = array<i64: 1, 2, 3>}
  return
}

// CHECK-LABEL: func.func @test_permutation()
// CHECK-NOT: scf.for
//       CHECK:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} 200 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 1 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 1 : i32} 200 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 2 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 2 : i32} 200 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 3 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 3 : i32} 200 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 4 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 4 : i32} 200 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} 100 : i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} 200 : i32
//  CHECK-NEXT:   return
func.func @test_permutation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  // Loop with static bounds and permutation
  scf.for %i = %c0 to %c6 step %c1 {
    %a = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32} 100 : i32
    %b = arith.constant {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} 200 : i32
  } {sched.dims = array<i64: 1, 2, 3>}
  return
}

// Check that SSA values are properly connected when cloning operations.
// Verify SSA use-def chains are correctly maintained.
// CHECK-LABEL: func.func @test_ssa_chains()
// All producers i32 first
// CHECK-DAG: %[[VAL_0:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : () -> i32
// CHECK-DAG: %[[VAL_1:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 1 : i32} : () -> i32
// CHECK-DAG: %[[VAL_2:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 2 : i32} : () -> i32
// CHECK-DAG: %[[VAL_4:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 3 : i32} : () -> i32
// CHECK-DAG: %[[VAL_6:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 4 : i32} : () -> i32
// CHECK-DAG: %[[VAL_8:[0-9]+]] = "test.producer"() {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : () -> i32
// All producers f32 second
// CHECK-DAG: %[[VAL_3:[0-9]+]] = "test.producer"(%[[VAL_0]]) {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 2 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_7:[0-9]+]] = "test.producer"(%[[VAL_1]]) {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 4 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_9:[0-9]+]] = "test.producer"(%[[VAL_2]]) {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 6 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_11:[0-9]+]] = "test.producer"(%[[VAL_4]]) {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 8 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_13:[0-9]+]] = "test.producer"(%[[VAL_6]]) {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 10 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_14:[0-9]+]] = "test.producer"(%[[VAL_8]]) {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 12 : i32} : (i32) -> f32
// All consumers last
// CHECK-DAG: %[[VAL_5:[0-9]+]] = "test.consumer"(%[[VAL_0]], %[[VAL_3]]) {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 3 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_10:[0-9]+]] = "test.consumer"(%[[VAL_1]], %[[VAL_7]]) {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 6 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_12:[0-9]+]] = "test.consumer"(%[[VAL_2]], %[[VAL_9]]) {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 9 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_15:[0-9]+]] = "test.consumer"(%[[VAL_4]], %[[VAL_11]]) {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 12 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_16:[0-9]+]] = "test.consumer"(%[[VAL_6]], %[[VAL_13]]) {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 15 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_17:[0-9]+]] = "test.consumer"(%[[VAL_8]], %[[VAL_14]]) {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 18 : i32} : (i32, f32) -> i8
func.func @test_ssa_chains() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %0 = "test.producer"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
    %1 = "test.producer"(%0) {sched.delay = 2 : i32, sched.rate = 2 : i32} : (i32) -> f32
    %2 = "test.consumer"(%0, %1) {sched.delay = 3 : i32, sched.rate = 3 : i32} : (i32, f32) -> i8
  } {sched.dims = array<i64: 1, 2, 3>}
  return
}

// Test that when consumer is shifted enough, we can still permute and keep SSA chains valid.
// Here, VAL_10 is the key synchronization point.
// Verify SSA use-def chains are correctly maintained with permutations.
// CHECK-LABEL: func.func @test_ssa_chains_with_permutation()
// All producers i32 checked first
// CHECK-DAG: %[[VAL_0:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : () -> i32
// CHECK-DAG: %[[VAL_1:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 1 : i32} : () -> i32
// CHECK-DAG: %[[VAL_2:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 2 : i32} : () -> i32
// CHECK-DAG: %[[VAL_4:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 3 : i32} : () -> i32
// CHECK-DAG: %[[VAL_5:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 4 : i32} : () -> i32
// CHECK-DAG: %[[VAL_7:[0-9]+]] = "test.producer_1"() {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : () -> i32
// Producers f32 checked second
// CHECK-DAG: %[[VAL_3:[0-9]+]] = "test.producer_2"(%[[VAL_0]]) {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 2 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_6:[0-9]+]] = "test.producer_2"(%[[VAL_1]]) {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 4 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_9:[0-9]+]] = "test.producer_2"(%[[VAL_2]]) {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 6 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_10:[0-9]+]] = "test.producer_2"(%[[VAL_4]]) {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 8 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_12:[0-9]+]] = "test.producer_2"(%[[VAL_5]]) {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 10 : i32} : (i32) -> f32
// CHECK-DAG: %[[VAL_14:[0-9]+]] = "test.producer_2"(%[[VAL_7]]) {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 12 : i32} : (i32) -> f32
// All consumers checked last
// CHECK-DAG: %[[VAL_8:[0-9]+]] = "test.consumer_permuted"(%[[VAL_0]], %[[VAL_3]]) {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 5 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_11:[0-9]+]] = "test.consumer_permuted"(%[[VAL_4]], %[[VAL_10]]) {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 8 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_13:[0-9]+]] = "test.consumer_permuted"(%[[VAL_1]], %[[VAL_6]]) {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 11 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_15:[0-9]+]] = "test.consumer_permuted"(%[[VAL_5]], %[[VAL_12]]) {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 14 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_16:[0-9]+]] = "test.consumer_permuted"(%[[VAL_2]], %[[VAL_9]]) {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 17 : i32} : (i32, f32) -> i8
// CHECK-DAG: %[[VAL_17:[0-9]+]] = "test.consumer_permuted"(%[[VAL_7]], %[[VAL_14]]) {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 20 : i32} : (i32, f32) -> i8
func.func @test_ssa_chains_with_permutation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %0 = "test.producer_1"() {sched.delay = 0 : i32, sched.rate = 1 : i32} : () -> i32
    %1 = "test.producer_2"(%0) {sched.delay = 2 : i32, sched.rate = 2 : i32} : (i32) -> f32
    %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 5 : i32, sched.rate = 3 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  } {sched.dims = array<i64: 1, 2, 3>}
  return
}


// CHECK-LABEL: func.func @test_ssa_chains_with_all_permutation()
//       CHECK:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : (i32, f32) -> i8
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 1 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 1 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 1 : i32} : (i32, f32) -> i8
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 2 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 2 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 2 : i32} : (i32, f32) -> i8
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 3 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 3 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 3 : i32} : (i32, f32) -> i8
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 4 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 4 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 4 : i32} : (i32, f32) -> i8
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : () -> i32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : (i32) -> f32
//  CHECK-NEXT:   unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : (i32, f32) -> i8
func.func @test_ssa_chains_with_all_permutation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %0 = "test.producer"()                {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : () -> i32
    %1 = "test.producer"(%0)              {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32) -> f32
    %2 = "test.consumer_permuted"(%0, %1) {sched.delay = 0 : i32, sched.rate = 1 : i32, sched.permutation = array<i32: 2, 0, 1>} : (i32, f32) -> i8
  } {sched.dims = array<i64: 1, 2, 3>}
  return
}

// Test case for instruction scheduling with loop induction variable and external values
// CHECK-LABEL: func.func @test_loop_values()
//       CHECK:   arith.constant 0 : index
//       CHECK:   arith.constant 1 : index
//       CHECK:   arith.constant 6 : index
//       CHECK:   %[[c42:.*]] = arith.constant 42 : i32
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[c2:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[c3:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[c4:.*]] = arith.constant 4 : index
//  CHECK-NEXT:   %[[c5:.*]] = arith.constant 5 : index
//
//  CHECK-DAG:   %[[add0:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 0 : i32} : i32
//  CHECK-DAG:   %[[add1:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 1 : i32} : i32
//  CHECK-DAG:   %[[add2:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 2 : i32} : i32
//  CHECK-DAG:   %[[add3:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 3 : i32} : i32
//  CHECK-DAG:   %[[add4:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 4 : i32} : i32
//  CHECK-DAG:   %[[add5:.*]] = arith.muli %[[c42]], %[[c42]] {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 5 : i32} : i32
//
//  CHECK-DAG:   %[[cast0:.*]] = arith.index_cast %[[c0]] {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 1 : i32} : index to i32
//  CHECK-DAG:   %[[cast1:.*]] = arith.index_cast %[[c1]] {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 2 : i32} : index to i32
//  CHECK-DAG:   %[[cast2:.*]] = arith.index_cast %[[c2]] {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 3 : i32} : index to i32
//  CHECK-DAG:   %[[cast3:.*]] = arith.index_cast %[[c3]] {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 4 : i32} : index to i32
//  CHECK-DAG:   %[[cast4:.*]] = arith.index_cast %[[c4]] {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 5 : i32} : index to i32
//  CHECK-DAG:   %[[cast5:.*]] = arith.index_cast %[[c5]] {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 6 : i32} : index to i32
//
//  CHECK-DAG:   arith.addi %[[add0]], %[[cast0]] {unroll.dims = array<i64: 0, 0, 0>, unroll.global_idx = 2 : i32} : i32
//  CHECK-DAG:   arith.addi %[[add1]], %[[cast1]] {unroll.dims = array<i64: 0, 0, 1>, unroll.global_idx = 4 : i32} : i32
//  CHECK-DAG:   arith.addi %[[add2]], %[[cast2]] {unroll.dims = array<i64: 0, 0, 2>, unroll.global_idx = 6 : i32} : i32
//  CHECK-DAG:   arith.addi %[[add3]], %[[cast3]] {unroll.dims = array<i64: 0, 1, 0>, unroll.global_idx = 8 : i32} : i32
//  CHECK-DAG:   arith.addi %[[add4]], %[[cast4]] {unroll.dims = array<i64: 0, 1, 1>, unroll.global_idx = 10 : i32} : i32
//  CHECK-DAG:   arith.addi %[[add5]], %[[cast5]] {unroll.dims = array<i64: 0, 1, 2>, unroll.global_idx = 12 : i32} : i32
func.func @test_loop_values() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %external_value = arith.constant 42 : i32

  scf.for %i = %c0 to %c6 step %c1 {
    %use_external = arith.muli %external_value, %external_value {sched.delay = 0 : i32, sched.rate = 1 : i32} : i32
    %idx_as_i32 = arith.index_cast %i {sched.delay = 1 : i32, sched.rate = 1 : i32} : index to i32
    %combined = arith.addi %use_external, %idx_as_i32 {sched.delay = 2 : i32, sched.rate = 2 : i32} : i32
  } {sched.dims = array<i64: 1, 2, 3>}

  return
}

// Only check that we properly unroll scf.if and that IR is valid in this test.
// Previous tests + composability ensure proper scheduling and SSA-chains are valid.
// CHECK-LABEL: func.func @test_scf_if_regions
// CHECK-COUNT-4:   scf.if
func.func @test_scf_if_regions() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index

  scf.for %i = %c0 to %c4 step %c1 {
    // Compute a condition based on iteration index
    %is_even = arith.remui %i, %c2 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index
    %cond = arith.cmpi eq, %is_even, %c0 {sched.delay = 1 : i32, sched.rate = 1 : i32} : index

    // Use scf.if with regions that depend on loop values
    %result = scf.if %cond -> (index) {
      // Even iteration: multiply by 2
      %doubled = arith.muli %i, %c2 {sched.delay = 2 : i32, sched.rate = 2 : i32} : index
      scf.yield %doubled : index
    } else {
      // Odd iteration: add 10
      %c10 = arith.constant 10 : index
      %added = arith.addi %i, %c10 {sched.delay = 2 : i32, sched.rate = 2 : i32} : index
      scf.yield %added : index
    } {sched.delay = 2 : i32, sched.rate = 2 : i32}

    // Use the result from the if statement
    %final = arith.muli %result, %result {sched.delay = 3 : i32, sched.rate = 3 : i32} : index
  } {sched.dims = array<i64: 2, 2>}

  return
}

// Test that loops with i32 induction variables create i32 constants during unrolling
// CHECK-LABEL: func.func @test_i32_loop_induction_variable
// CHECK-NOT: scf.for
// CHECK: arith.constant 0 : i32
// CHECK: arith.constant 1 : i32
// CHECK: arith.constant 2 : i32
func.func @test_i32_loop_induction_variable() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32

  scf.for unsigned %i = %c0 to %c3 step %c1 : i32 {
    %use_i = arith.muli %i, %c1 {sched.delay = 0 : i32, sched.rate = 1 : i32} : i32
  } {sched.dims = array<i64: 3>}

  return
}
