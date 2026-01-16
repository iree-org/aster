// Unit test kernels for indexing.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @test_indexing target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  func.func private @lane_id() -> index
  func.func private @wave_id() -> index
  func.func private @wave_count() -> index
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @block_id_x_delinearize_2d(index, index) -> (index, index)
  func.func private @tiled_grid_partition_2d(index, index, index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @mfma_index_16x16_helper() -> (index, index)
  func.func private @mfma_index_A_16x16xf16() -> (index, index)
  func.func private @mfma_index_B_16x16xf16() -> (index, index)
  func.func private @mfma_index_C_16x16xf32() -> (index, index)
  func.func private @swizzled_mfma_index_A_16x16xf16() -> (index, index)
  func.func private @swizzled_mfma_index_B_16x16xf16() -> (index, index)
  func.func private @swizzled_mfma_index_C_16x16xf32() -> (index, index)
  func.func private @index_bxmxnxk_16x16x16_f16f16f32(index, index, index, index, index, index, index, index, index) -> index

  //===--------------------------------------------------------------------===//
  // Helper: store i32 to global at thread index
  //===--------------------------------------------------------------------===//
  func.func private @store_at_tid(%value: i32, %ptr: !sx2, %index_offset: index) {
    %tid = gpu.thread_id x
    %value_vgpr = lsir.to_reg %value : i32 -> !v

    %offset_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 4 + index_offset)>()[%tid, %index_offset]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !v

    %c0 = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dword data %value_vgpr addr %ptr offset d(%offset_vgpr) + c(%c0) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return
  }

  // Store two i32 values at thread index (for functions returning two values)
  func.func private @store_pair_at_tid(%v0: i32, %v1: i32, %ptr: !sx2, %index_offset: index) {
    %tid = gpu.thread_id x

    %c0_pair = arith.constant 0 : i32

    // Store v0 at tid * 8, v1 at tid * 8 + 4
    %offset0_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 8 + index_offset)>()[%tid, %index_offset]
    %offset0 = arith.index_cast %offset0_index : index to i32
    %offset0_vgpr = lsir.to_reg %offset0 : i32 -> !v
    %v0_vgpr = lsir.to_reg %v0 : i32 -> !v
    %tok_store0 = amdgcn.store global_store_dword data %v0_vgpr addr %ptr offset d(%offset0_vgpr) + c(%c0_pair) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %offset1_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 8 + 4 + index_offset)>()[%tid, %index_offset]
    %offset1 = arith.index_cast %offset1_index : index to i32
    %offset1_vgpr = lsir.to_reg %offset1 : i32 -> !v
    %v1_vgpr = lsir.to_reg %v1 : i32 -> !v
    %tok_store1 = amdgcn.store global_store_dword data %v1_vgpr addr %ptr offset d(%offset1_vgpr) + c(%c0_pair) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // Test kernels - one per function
  //===--------------------------------------------------------------------===//

  // Test @lane_id: each thread writes its lane_id (0..63 repeating)
  amdgcn.kernel @test_lane_id arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %lane = func.call @lane_id() : () -> index
    %lane_i32 = arith.index_cast %lane : index to i32
    %c0 = arith.constant 0 : index
    func.call @store_at_tid(%lane_i32, %out_ptr, %c0) : (i32, !sx2, index) -> ()

    amdgcn.end_kernel
  }

  // Test @wave_id: each thread writes its wave_id
  amdgcn.kernel @test_wave_id arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %wave = func.call @wave_id() : () -> index
    %wave_i32 = arith.index_cast %wave : index to i32
    %c0 = arith.constant 0 : index
    func.call @store_at_tid(%wave_i32, %out_ptr, %c0) : (i32, !sx2, index) -> ()

    amdgcn.end_kernel
  }

  // Test @wave_count: each thread writes the wave_count
  amdgcn.kernel @test_wave_count arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %count = func.call @wave_count() : () -> index
    %count_i32 = arith.index_cast %count : index to i32
    %c0 = arith.constant 0 : index
    func.call @store_at_tid(%count_i32, %out_ptr, %c0) : (i32, !sx2, index) -> ()

    amdgcn.end_kernel
  }

  // Test @lane_delinearize_2d: partition 64 lanes by 8x8
  amdgcn.kernel @test_lane_delinearize_2d arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c8 = arith.constant 8 : index
    %i, %j = func.call @lane_delinearize_2d(%c8, %c8) : (index, index) -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    %c0 = arith.constant 0 : index
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()

    amdgcn.end_kernel
  }

  // Test @block_id_x_delinearize_2d: partition block_id_x by 2x4
  amdgcn.kernel @test_block_id_x_delinearize_2d arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %i, %j = func.call @block_id_x_delinearize_2d(%c2, %c4) : (index, index) -> (index, index)

    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32

    // Shift out_ptr by block_idx * 64 * (2x4) bytes.
    %block_idx = gpu.block_id x
    %offset = affine.apply affine_map<()[block_idx] -> (block_idx * 64 * (2 * 4))>()[%block_idx]
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %offset) : (i32, i32, !sx2, index) -> ()

    amdgcn.end_kernel
  }

  // Test @tiled_grid_partition_2d: partition for tiled problems
  // M=64, N=64, M_TILE=32, N_TILE=32 -> 2x2 tiles
  amdgcn.kernel @test_tiled_grid_partition_2d arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %i, %j = func.call @tiled_grid_partition_2d(%c64, %c64, %c32, %c32)
      : (index, index, index, index) -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32

    // Shift out_ptr by block_idx * 64 * (2x4) bytes.
    %block_idx = gpu.block_id x
    %offset = affine.apply affine_map<()[block_idx] -> (block_idx * 64 * (2 * 4))>()[%block_idx]
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %offset) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @matrix_offset: compute byte offset for 2D matrix access
  // Uses i=thread_id/8, j=thread_id%8, stride=64 (16*4 bytes), elt_size=4
  amdgcn.kernel @test_matrix_offset arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)
    %c4 = arith.constant 4 : index
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]
    %off_vgpr = func.call @matrix_offset(%i, %j, %c64, %c4)
      : (index, index, index, index) -> !v

    // Store the offset at thread position
    %out_offset_index = affine.apply affine_map<()[tid] -> (tid * 4)>()[%tid]
    %out_offset = arith.index_cast %out_offset_index : index to i32
    %out_offset_vgpr = lsir.to_reg %out_offset : i32 -> !v
    %c0_mo = arith.constant 0 : i32
    %tok1 = amdgcn.store global_store_dword data %off_vgpr addr %out_ptr offset d(%out_offset_vgpr) + c(%c0_mo) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // Test @tiled_matrix_offset: compute byte offset for tiled 2D matrix access
  // Uses i=0, j=0, ii=tid/8, jj=tid%8, stride=64 (16*4 bytes), elt_size=4
  amdgcn.kernel @test_tiled_matrix_offset arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %tid = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)
    %c4 = arith.constant 4 : index
    %ii = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %jj = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]
    %off_vgpr = func.call @tiled_matrix_offset(%c0, %c0, %ii, %jj, %c64, %c4)
      : (index, index, index, index, index, index) -> !v

    // Store the offset at thread position
    %out_offset_index = affine.apply affine_map<()[tid] -> (tid * 4)>()[%tid]
    %out_offset = arith.index_cast %out_offset_index : index to i32
    %out_offset_vgpr = lsir.to_reg %out_offset : i32 -> !v
    %c0_tmo = arith.constant 0 : i32
    %tok2 = amdgcn.store global_store_dword data %off_vgpr addr %out_ptr offset d(%out_offset_vgpr) + c(%c0_tmo) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // Test @tiledx2_matrix_offset: compute byte offset for twice-tiled 2D matrix access
  // Uses i=0, j=0, ii=0, jj=0, iii=tid/8, jjj=tid%8, stride=64 (16*4 bytes), elt_size=4
  amdgcn.kernel @test_tiledx2_matrix_offset arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %tid = gpu.thread_id x
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)
    %c4 = arith.constant 4 : index
    %iii = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %jjj = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]
    %off_vgpr = func.call @tiledx2_matrix_offset(%c0, %c0, %c0, %c0, %iii, %jjj, %c64, %c4)
      : (index, index, index, index, index, index, index, index) -> !v

    // Store the offset at thread position
    %out_offset_index = affine.apply affine_map<()[tid] -> (tid * 4)>()[%tid]
    %out_offset = arith.index_cast %out_offset_index : index to i32
    %out_offset_vgpr = lsir.to_reg %out_offset : i32 -> !v
    %c0_tx2 = arith.constant 0 : i32
    %tok3 = amdgcn.store global_store_dword data %off_vgpr addr %out_ptr offset d(%out_offset_vgpr) + c(%c0_tx2) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // Test @mfma_index_16x16_helper: returns (4 * (lane_id / 16), lane_id mod 16)
  amdgcn.kernel @test_mfma_index_16x16_helper arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @mfma_index_16x16_helper() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @mfma_index_A_16x16xf16: MFMA indexing for A fragment (swapped from helper)
  amdgcn.kernel @test_mfma_index_A_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @mfma_index_A_16x16xf16() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @mfma_index_B_16x16xf16: MFMA indexing for B fragment
  amdgcn.kernel @test_mfma_index_B_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @mfma_index_B_16x16xf16() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @mfma_index_C_16x16xf32: MFMA indexing for C fragment
  amdgcn.kernel @test_mfma_index_C_16x16xf32 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @mfma_index_C_16x16xf32() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @swizzled_mfma_index_A_16x16xf16: swizzled MFMA indexing for A fragment
  amdgcn.kernel @test_swizzled_mfma_index_A_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @swizzled_mfma_index_A_16x16xf16() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @swizzled_mfma_index_B_16x16xf16: swizzled MFMA indexing for B fragment
  amdgcn.kernel @test_swizzled_mfma_index_B_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @swizzled_mfma_index_B_16x16xf16() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @swizzled_mfma_index_C_16x16xf32: swizzled MFMA indexing for C fragment
  amdgcn.kernel @test_swizzled_mfma_index_C_16x16xf32 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %i, %j = func.call @swizzled_mfma_index_C_16x16xf32() : () -> (index, index)
    %i_i32 = arith.index_cast %i : index to i32
    %j_i32 = arith.index_cast %j : index to i32
    func.call @store_pair_at_tid(%i_i32, %j_i32, %out_ptr, %c0) : (i32, i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }

  // Test @index_bxmxnxk_16x16x16_f16f16f32: MFMA-style tiled indexing
  // Uses fixed values to test the formula
  amdgcn.kernel @test_index_bxmxnxk arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %c0 = arith.constant 0 : index
    %bidx = gpu.block_id x
    %tidx = gpu.thread_id x
    %bdimx = gpu.block_dim x
    // Fixed test parameters
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index   // lane_stride
    %c16 = arith.constant 16 : index // tile_size
    %c2 = arith.constant 2 : index   // szI, szJ
    %offset = func.call @index_bxmxnxk_16x16x16_f16f16f32(
      %bidx, %tidx, %c0, %c0, %c2, %c2, %bdimx, %c16, %c4)
      : (index, index, index, index, index, index, index, index, index) -> index
    %offset_i32 = arith.index_cast %offset : index to i32
    func.call @store_at_tid(%offset_i32, %out_ptr, %c0) : (i32, !sx2, index) -> ()
    amdgcn.end_kernel
  }
}
