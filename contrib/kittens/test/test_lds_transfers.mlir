// Unit test for LDS transfer primitives
// RUN: aster-opt %s | FileCheck %s

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.module @test_lds_transfers target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/lds_16x16_f16.mlir
  func.func private @alloc_lds_1buffer_padded() -> (index, index)

  // From kittens/lds_16x16_f16.mlir
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_f16(index) -> !future_lds_read
  func.func private @load_lds_B_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16
  func.func private @store_lds_A_f16(!rt_A_f16, index) -> !lds_write_token

  // Test Global -> LDS transfer
  // CHECK-LABEL: @test_load_global_to_lds
  func.func @test_load_global_to_lds(%ptr: !sx2) {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    // Tile position and stride
    %m = arith.constant 0 : index
    %n = arith.constant 0 : index
    %stride = arith.constant 64 : index  // 32 * 2 bytes for f16

    // CHECK: call @load_global_to_lds_f16
    %tok = func.call @load_global_to_lds_f16(%A_base, %ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> !lds_write_token
    amdgcn.wait deps %tok : !lds_write_token

    return
  }

  // Test LDS -> Register transfer (A tile)
  // CHECK-LABEL: @test_get_lds_A
  func.func @test_get_lds_A() -> !rt_A_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    // CHECK: call @load_lds_A_f16
    %future = func.call @load_lds_A_f16(%A_base)
        : (index) -> !future_lds_read
    // CHECK: call @get_lds_A_f16
    %tile = func.call @get_lds_A_f16(%future)
        : (!future_lds_read) -> !rt_A_f16

    return %tile : !rt_A_f16
  }

  // Test LDS -> Register transfer (B tile)
  // CHECK-LABEL: @test_get_lds_B
  func.func @test_get_lds_B() -> !rt_B_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    // CHECK: call @load_lds_B_f16
    %future = func.call @load_lds_B_f16(%B_base)
        : (index) -> !future_lds_read
    // CHECK: call @get_lds_B_f16
    %tile = func.call @get_lds_B_f16(%future)
        : (!future_lds_read) -> !rt_B_f16

    return %tile : !rt_B_f16
  }

  // Test Register -> LDS transfer
  // CHECK-LABEL: @test_store_register_to_lds
  func.func @test_store_register_to_lds(%tile: !rt_A_f16) {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    // CHECK: call @store_lds_A_f16
    %tok = func.call @store_lds_A_f16(%tile, %A_base)
        : (!rt_A_f16, index) -> !lds_write_token
    amdgcn.wait deps %tok : !lds_write_token

    return
  }

  // Integration test: Full round-trip (Global -> LDS -> Register -> LDS)
  // CHECK-LABEL: @test_roundtrip
  func.func @test_roundtrip(%ptr: !sx2) -> !rt_A_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    %m = arith.constant 0 : index
    %n = arith.constant 0 : index
    %stride = arith.constant 64 : index

    // Step 1: Global -> LDS
    // CHECK: call @load_global_to_lds_f16
    %tok = func.call @load_global_to_lds_f16(%A_base, %ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> !lds_write_token
    amdgcn.wait deps %tok : !lds_write_token

    // Step 2: LDS -> Register
    // CHECK: call @load_lds_A_f16
    %future = func.call @load_lds_A_f16(%A_base)
        : (index) -> !future_lds_read
    // CHECK: call @get_lds_A_f16
    %tile = func.call @get_lds_A_f16(%future)
        : (!future_lds_read) -> !rt_A_f16

    // Step 3: Register -> LDS (write to B buffer)
    // CHECK: call @store_lds_A_f16
    %tok2 = func.call @store_lds_A_f16(%tile, %B_base)
        : (!rt_A_f16, index) -> !lds_write_token
    amdgcn.wait deps %tok2 : !lds_write_token

    // Step 4: LDS -> Register (read back from B buffer)
    // CHECK: call @load_lds_A_f16
    %future2 = func.call @load_lds_A_f16(%B_base)
        : (index) -> !future_lds_read
    // CHECK: call @get_lds_A_f16
    %tile2 = func.call @get_lds_A_f16(%future2)
        : (!future_lds_read) -> !rt_A_f16

    return %tile2 : !rt_A_f16
  }
}
