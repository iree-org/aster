// RUN: aster-opt --aster-op-scheduling="test-only=true" --allow-unregistered-dialect --mlir-disable-threading %s | FileCheck %s

// Test scheduling of operations with nested regions (scf.if).
// Operations with regions are scheduled as atomic units - the entire scf.if
// is cloned together with its nested operations.

// CHECK-LABEL: func.func @test_scf_if_basic
// CHECK-NOT: scf.for
// CHECK-COUNT-4: scf.if
func.func @test_scf_if_basic() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c4 step %c1 {
    // scf.if uses only values defined outside the loop - always safe
    scf.if %true {
      %x = arith.constant 42 : i32
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 4>}

  return
}

// Test that scf.if can use values from earlier operations in the same iteration
// when those operations have delay <= scf.if's delay.
// CHECK-LABEL: func.func @test_scf_if_uses_earlier_op_same_delay
// CHECK-NOT: scf.for
// CHECK-COUNT-2: scf.if
func.func @test_scf_if_uses_earlier_op_same_delay() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  scf.for %i = %c0 to %c2 step %c1 {
    // Producer has delay=0, consumer (scf.if) has delay=0
    // This is safe because producer fires at same time or before
    %cond = arith.cmpi eq, %i, %c0 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index
    scf.if %cond {
      // Nested ops can use %cond since it's the scf.if's operand
      %x = arith.constant 1 : i32
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test that nested region can use values produced earlier with lower delay
// CHECK-LABEL: func.func @test_scf_if_nested_uses_earlier_producer
// CHECK-NOT: scf.for
// CHECK-COUNT-2: scf.if
func.func @test_scf_if_nested_uses_earlier_producer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  scf.for %i = %c0 to %c2 step %c1 {
    // Producer at delay=0
    %val = arith.addi %i, %c1 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index
    %cond = arith.cmpi sgt, %val, %c0 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index

    // scf.if at delay=2 - producer is already available
    scf.if %cond {
      // Nested op uses %val which is defined at delay=0 <= scf.if delay=2
      %use_val = arith.muli %val, %c2 : index
    } {sched.delay = 2 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test that nested operations inside scf.if can reference values defined
// within the same scf.if region (not external) - this should always work.
// CHECK-LABEL: func.func @test_scf_if_internal_dependencies
// CHECK-NOT: scf.for
// CHECK-COUNT-4: scf.if
func.func @test_scf_if_internal_dependencies() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %true {
      // Internal dependencies within the region - these are cloned together
      %a = arith.constant 1 : i32
      %b = arith.constant 2 : i32
      %c = arith.addi %a, %b : i32  // Uses %a and %b from same region
      %d = arith.muli %c, %a : i32  // Uses %c and %a from same region
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 4>}

  return
}

// Test scf.if with else region
// CHECK-LABEL: func.func @test_scf_if_with_else
// CHECK-NOT: scf.for
// CHECK-COUNT-3: scf.if
func.func @test_scf_if_with_else() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index

  scf.for %i = %c0 to %c3 step %c1 {
    %is_even = arith.remui %i, %c2 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index
    %cond = arith.cmpi eq, %is_even, %c0 {sched.delay = 0 : i32, sched.rate = 1 : i32} : index

    %result = scf.if %cond -> (index) {
      %doubled = arith.muli %i, %c2 : index
      scf.yield %doubled : index
    } else {
      %c10 = arith.constant 10 : index
      %added = arith.addi %i, %c10 : index
      scf.yield %added : index
    } {sched.delay = 1 : i32, sched.rate = 1 : i32}

    // Use result outside the if
    %final = arith.muli %result, %c2 {sched.delay = 2 : i32, sched.rate = 1 : i32} : index
  } {sched.dims = array<i64: 3>}

  return
}

// Test multiple scf.if operations in the same loop
// CHECK-LABEL: func.func @test_multiple_scf_if
// CHECK-NOT: scf.for
// CHECK-COUNT-4: scf.if
func.func @test_multiple_scf_if() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  %false = arith.constant false

  scf.for %i = %c0 to %c2 step %c1 {
    // First scf.if
    scf.if %true {
      %x = arith.constant 1 : i32
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}

    // Second scf.if with different schedule
    scf.if %false {
      %y = arith.constant 2 : i32
    } {sched.delay = 1 : i32, sched.rate = 1 : i32}
  } {sched.dims = array<i64: 2>}

  return
}

// Test that scf.if result can be used by later operations
// CHECK-LABEL: func.func @test_scf_if_result_used_later
// CHECK-NOT: scf.for
func.func @test_scf_if_result_used_later() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true

  scf.for %i = %c0 to %c2 step %c1 {
    %if_result = scf.if %true -> (i32) {
      %x = arith.constant 42 : i32
      scf.yield %x : i32
    } else {
      %y = arith.constant 0 : i32
      scf.yield %y : i32
    } {sched.delay = 0 : i32, sched.rate = 1 : i32}

    // Use scf.if result - scheduled after scf.if
    %doubled = arith.muli %if_result, %if_result {sched.delay = 1 : i32, sched.rate = 1 : i32} : i32
  } {sched.dims = array<i64: 2>}

  return
}
