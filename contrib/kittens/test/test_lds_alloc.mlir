// Unit test for LDS allocation and addressing primitives
// RUN: aster-opt %s | FileCheck %s

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!rt_A_f16 = !vx2

amdgcn.module @test_lds_alloc target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/lds_16x16_f16.mlir
  func.func private @alloc_lds_1buffer_padded() -> (index, index)
  func.func private @alloc_lds_2buffer_padded() -> (index, index, index, index)
  func.func private @alloc_lds_3buffer_padded() -> (index, index, index, index, index, index)
  func.func private @lds_element_offset_padded(index, index, index) -> index
  func.func private @thread_lds_slice() -> (index, index)

  // Test 1-buffer allocation
  // CHECK-LABEL: @test_alloc_1buffer
  func.func @test_alloc_1buffer() -> (index, index) {
    // CHECK: call @alloc_lds_1buffer_padded
    %A0, %B0 = func.call @alloc_lds_1buffer_padded() : () -> (index, index)
    // CHECK: return
    return %A0, %B0 : index, index
  }

  // Test 2-buffer allocation
  // CHECK-LABEL: @test_alloc_2buffer
  func.func @test_alloc_2buffer() -> (index, index, index, index) {
    // CHECK: call @alloc_lds_2buffer_padded
    %A0, %B0, %A1, %B1 = func.call @alloc_lds_2buffer_padded()
        : () -> (index, index, index, index)
    // CHECK: return
    return %A0, %B0, %A1, %B1 : index, index, index, index
  }

  // Test 3-buffer allocation
  // CHECK-LABEL: @test_alloc_3buffer
  func.func @test_alloc_3buffer() -> (index, index, index, index, index, index) {
    // CHECK: call @alloc_lds_3buffer_padded
    %A0, %B0, %A1, %B1, %A2, %B2 = func.call @alloc_lds_3buffer_padded()
        : () -> (index, index, index, index, index, index)
    // CHECK: return
    return %A0, %B0, %A1, %B1, %A2, %B2 : index, index, index, index, index, index
  }

  // Test element offset calculation
  // CHECK-LABEL: @test_element_offset
  func.func @test_element_offset() -> (index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %tile_base = arith.constant 0 : index

    // Element at (0, 0) - should be tile_base + 0
    // CHECK: call @lds_element_offset_padded
    %off_0_0 = func.call @lds_element_offset_padded(%tile_base, %c0, %c0)
        : (index, index, index) -> index

    // Element at (1, 0) - should be tile_base + 34 (row stride = 17*2 bytes)
    // CHECK: call @lds_element_offset_padded
    %off_1_0 = func.call @lds_element_offset_padded(%tile_base, %c1, %c0)
        : (index, index, index) -> index

    // Element at (15, 15) - should be tile_base + 15*34 + 15*2 = 540
    // CHECK: call @lds_element_offset_padded
    %off_15_15 = func.call @lds_element_offset_padded(%tile_base, %c15, %c15)
        : (index, index, index) -> index

    // CHECK: return
    return %off_0_0, %off_1_0, %off_15_15 : index, index, index
  }

  // Test thread-to-element mapping
  // CHECK-LABEL: @test_thread_slice
  func.func @test_thread_slice() -> (index, index) {
    // CHECK: call @thread_lds_slice
    %row, %col = func.call @thread_lds_slice() : () -> (index, index)
    // CHECK: return
    return %row, %col : index, index
  }

  // Integration test: Allocate 2-buffer and compute element offsets
  // CHECK-LABEL: @test_integration_2buffer
  func.func @test_integration_2buffer() -> (index, index) {
    // Allocate 2-buffer LDS
    // CHECK: call @alloc_lds_2buffer_padded
    %A0, %B0, %A1, %B1 = func.call @alloc_lds_2buffer_padded()
        : () -> (index, index, index, index)

    // Compute offset for element (5, 10) in A[0]
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    // CHECK: call @lds_element_offset_padded
    %off_A0 = func.call @lds_element_offset_padded(%A0, %c5, %c10)
        : (index, index, index) -> index

    // Compute offset for element (7, 3) in B[1]
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    // CHECK: call @lds_element_offset_padded
    %off_B1 = func.call @lds_element_offset_padded(%B1, %c7, %c3)
        : (index, index, index) -> index

    // CHECK: return
    return %off_A0, %off_B1 : index, index
  }
}
