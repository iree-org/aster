// RUN: aster-opt %s --verify-roundtrip

// Test sched.unit with inputs, results, and stage.
// CHECK-LABEL: func.func @test_unit_simple
func.func @test_unit_simple(%arg0: i32, %arg1: f32) -> (i32, f32) {
  %0, %1 = sched.unit(%arg0, %arg1 : i32, f32) -> (i32, f32) stage 0 {
  ^bb0(%x: i32, %y: f32):
    sched.yield %x, %y : i32, f32
  }
  return %0, %1 : i32, f32
}

// Test sched.unit with barrier attribute.
// CHECK-LABEL: func.func @test_unit_barrier
func.func @test_unit_barrier(%arg0: index) -> index {
  %0 = sched.unit(%arg0 : index) -> (index) stage 1 barrier {
  ^bb0(%x: index):
    sched.yield %x : index
  }
  return %0 : index
}

// Test sched.unit with no inputs or results.
// CHECK-LABEL: func.func @test_unit_no_io
func.func @test_unit_no_io() {
  sched.unit stage 0 {
    sched.yield
  }
  return
}

// Test sched.loop_resource with all four regions inside an scf.for.
// CHECK-LABEL: func.func @test_loop_resource_full
func.func @test_loop_resource_full() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %token = sched.loop_resource -> i32 {
      %buf = arith.constant 42 : i32
      sched.yield %buf : i32
    } deallocate {
    ^bb0(%buf: i32):
      sched.yield
    } forward {
    ^bb0(%buf: i32):
      sched.yield %buf : i32
    } fence {
    ^bb0(%buf: i32):
      sched.yield
    }
  }
  return
}

// Test sched.loop_resource with only the required regions.
// CHECK-LABEL: func.func @test_loop_resource_minimal
func.func @test_loop_resource_minimal() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %token = sched.loop_resource -> i32 {
      %buf = arith.constant 0 : i32
      sched.yield %buf : i32
    } forward {
    ^bb0(%buf: i32):
      sched.yield %buf : i32
    }
  }
  return
}
