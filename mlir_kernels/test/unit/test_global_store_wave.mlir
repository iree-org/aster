// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  func.func private @init_vgprx4(i32) -> !vx4
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!sx2, index, index, index, index, index, index, index)
  func.func private @simple_global_store_wave_16x16xf16_wait(!vx2, !sx2, index, index, index)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(index, index, index, index, !sx2, index, index, index)
  // copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(!sx2, index, index, index, index, index, index, index)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(index, index, index, index, index, !vx2) -> ()
  func.func private @store_to_global_dword_wait(!v, !sx2, index, index, index)
  func.func private @store_to_global_dwordx2_wait(!vx2, !sx2, index, index, index)
  func.func private @store_to_global_dwordx3_wait(!vx3, !sx2, index, index, index)
  func.func private @store_to_global_dwordx4_wait(!vx4, !sx2, index, index, index)
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(index, index, index, index, i1) -> !vx2
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(index, index, index, index) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !sx2, index, index, index, index, index, i1)
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index, index, memref<?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(index, index, index, index, index, index, memref<?x!vx2>)
  func.func private @simple_global_load_wave_16x16xf16_wait(!sx2, index, index, index) -> !vx2
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, index, index, index, index)
  func.func private @simple_lds_read_wave_16x16xf16_wait(index, index, index, index) -> !vx2
  // simple-multi-tile-copies.mlir
  func.func private @simple_maybe_lds_write_multi_tile(index, index, index, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)
  // multi-tile-copies.mlir
  func.func private @simple_maybe_global_load_multi_tile(index, index, index, index, index, index, index, index, index, !sx2, index, index, index, memref<?x?x!vx2>)
  func.func private @maybe_global_load_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, !sx2, index, index, index, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)

  //===--------------------------------------------------------------------===//
  // Global store
  //===--------------------------------------------------------------------===//

  // Test @store_to_global_dword_wait: store a dword to global memory
  // Each thread stores (tid * 100) at position (tid/8, tid%8) in a 16-wide matrix
  amdgcn.kernel @test_store_to_global_dword_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)

    // Compute i, j from tid
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]

    // Compute value to store: tid * 100
    %value_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %value_i32 = arith.index_cast %value_idx : index to i32
    %value = lsir.to_reg %value_i32 : i32 -> !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    func.call @store_to_global_dword_wait(%value, %out_ptr, %c0, %tid, %c0)
      : (!v, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx2_wait: store 8 bytes to global memory
  // Each thread stores 2 dwords at position (tid/4, tid%4) in an 8-wide matrix
  amdgcn.kernel @test_store_to_global_dwordx2_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index // stride in bytes (8 elements * 8 bytes)

    // Compute i, j from tid (each thread writes 2 dwords = 1 column)
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 4)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 4)>()[%tid]

    // Compute values to store: [tid * 100, tid * 100 + 1]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1 : !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    func.call @store_to_global_dwordx2_wait(%value, %out_ptr, %c0, %tid, %c0)
      : (!vx2, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx3_wait: store 12 bytes to global memory
  // Each thread stores 3 dwords at linear position tid
  amdgcn.kernel @test_store_to_global_dwordx3_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x

    // Compute values to store: [tid * 100, tid * 100 + 1, tid * 100 + 2]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v2_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 2)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v2_i32 = arith.index_cast %v2_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %v2 = lsir.to_reg %v2_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1, %v2 : !v, !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    func.call @store_to_global_dwordx3_wait(%value, %out_ptr, %c0, %tid, %c0)
      : (!vx3, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx4_wait: store 16 bytes to global memory
  // Each thread stores 4 dwords at position (tid/4, tid%4) in a 4-wide matrix
  amdgcn.kernel @test_store_to_global_dwordx4_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x

    // Compute values to store: [tid * 100, tid * 100 + 1, tid * 100 + 2, tid * 100 + 3]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v2_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 2)>()[%tid]
    %v3_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 3)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v2_i32 = arith.index_cast %v2_idx : index to i32
    %v3_i32 = arith.index_cast %v3_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %v2 = lsir.to_reg %v2_i32 : i32 -> !v
    %v3 = lsir.to_reg %v3_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1, %v2, %v3 : !v, !v, !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    func.call @store_to_global_dwordx4_wait(%value, %out_ptr, %c0, %tid, %c0)
      : (!vx4, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

}
