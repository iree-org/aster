// Test LDS roundtrip with XOR swizzle addressing: Global -> LDS -> Register -> Global
// Verifies data survives the XOR-swizzled LDS path (same as padded roundtrip
// but using @lds_element_offset_xor_swizzle for bank conflict avoidance).

!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!rt_A_f16 = !vx2
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.module @test_kittens_lds_roundtrip_xor_swizzle target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @store_A_f16(!rt_A_f16, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir (XOR swizzle variants)
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16

  // From kittens/lds_16x16_f16.mlir (XOR swizzle variant)
  func.func private @alloc_lds_1buffer() -> (index, index)

  // Convenience wrapper: Global -> LDS (XOR swizzle) -> Register in one call.
  // Single-buffer, no latency hiding -- for testing only.
  func.func private @load_global_to_register_A_via_lds_xor_swizzle_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !rt_A_f16 {
    %tok = func.call @load_global_to_lds_xor_swizzle_f16(%lds_tile_base, %global_ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> !lds_write_token
    amdgcn.wait deps %tok : !lds_write_token
    // When we have multi-waves we should also barrier but we'll need to be
    // careful about impacts on pipelining + async.
    //   amdgcn.sopp.sopp <s_barrier>

    %future = func.call @load_lds_A_xor_swizzle_f16(%lds_tile_base)
        : (index) -> !future_lds_read
    %result = func.call @get_lds_A_f16(%future)
        : (!future_lds_read) -> !rt_A_f16
    return %result : !rt_A_f16
  }

  // Load 16x16 f16 tile: Global -> LDS (XOR swizzle) -> Register, then store back.
  // Input: 16x16 f16 matrix (row-major, stride = 32 bytes)
  // Output: 16x16 f16 matrix (should match input)
  amdgcn.kernel @test_lds_roundtrip_xor_swizzle arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 1024 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %stride = arith.constant 32 : index  // 16 * 2 bytes

    %lds_A, %lds_B = func.call @alloc_lds_1buffer() : () -> (index, index)

    // Global -> LDS (XOR swizzle) -> Register
    %tile = func.call @load_global_to_register_A_via_lds_xor_swizzle_f16(%lds_A, %in_ptr, %c0, %c0, %stride)
        : (index, !sx2, index, index, index) -> !rt_A_f16

    // Register -> Global (direct store)
    %store_tok = func.call @store_A_f16(%tile, %out_ptr, %c0, %c0, %stride)
        : (!rt_A_f16, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
