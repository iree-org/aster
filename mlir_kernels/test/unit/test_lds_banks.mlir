!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

amdgcn.module @test_indexing target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // indexing.mlir
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @lds_banks_for_transfer(index, index) -> (index, index, index, index, index, index, index, index)
  // copies.mlir
  func.func private @store_to_global_dwordx4_wait(!vx4, !sx2, index, index, index)

  //===--------------------------------------------------------------------===//
  // LDS bank debugging kernels
  //===--------------------------------------------------------------------===//

  // Test LDS banks for NON-swizzled MFMA A matrix pattern (for comparison).
  // Shows what banks would be accessed without swizzling.
  amdgcn.kernel @test_lds_banks_A_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32,
                 block_dims = array<i32: {{NUM_THREADS}}, 1, 1>,
                 grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>}
  {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %elt_size = arith.constant 2 : index   // f16 = 2 bytes
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[elt_size] -> (elt_size * 16)>()[%elt_size]

    // Get MFMA A indexing pattern WITHOUT swizzle
    %idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %idx ["i", "j"] : !index_pair -> index, index

    // Compute byte address in LDS directly (no swizzle)
    %desc = aster_utils.struct_create(%c0, %c0, %row, %col, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_vgpr = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    %off_i32 = lsir.from_reg %off_vgpr : !v -> i32
    %byte_address = arith.index_cast %off_i32 : i32 to index

    // Compute banks for an 8-byte access
    %transfer_size = arith.constant 8 : index  // dwordx2
    %b0, %b1, %b2, %b3, %b4_, %b5_, %b6_, %b7_ = func.call @lds_banks_for_transfer(%byte_address, %transfer_size)
      : (index, index) -> (index, index, index, index, index, index, index, index)


    // Store 4 banks as dwordx4 at tid position
    %tid = gpu.thread_id x
    %c16 = arith.constant 16 : index
    %b0_i32 = arith.index_cast %b0 : index to i32
    %b1_i32 = arith.index_cast %b1 : index to i32
    %b2_i32 = arith.index_cast %b2 : index to i32
    %b3_i32 = arith.index_cast %b3 : index to i32
    %b0_v = lsir.to_reg %b0_i32 : i32 -> !v
    %b1_v = lsir.to_reg %b1_i32 : i32 -> !v
    %b2_v = lsir.to_reg %b2_i32 : i32 -> !v
    %b3_v = lsir.to_reg %b3_i32 : i32 -> !v
    %data = amdgcn.make_register_range %b0_v, %b1_v, %b2_v, %b3_v : !v, !v, !v, !v
    func.call @store_to_global_dwordx4_wait(%data, %out_ptr, %c0, %tid, %c16)
      : (!vx4, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test LDS banks for swizzled MFMA A matrix pattern (16x16xf16).
  // Each thread outputs 4 banks accessed by a dwordx2 (8-byte) load at its swizzled address.
  // Output layout: tid * 16 bytes -> [bank0, bank1, bank2, bank3] (4 x i32)
  // This helps debug bank conflicts in swizzled LDS access patterns.
  amdgcn.kernel @test_lds_banks_swizzled_A_16x16xf16 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32,
                 block_dims = array<i32: {{NUM_THREADS}}, 1, 1>,
                 grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>}
  {
    %c0 = arith.constant 0 : index
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // LDS layout: 16 rows x N columns of f16 elements
    // LDS stride for a 16-column layout: 16 cols * 2 bytes = 32 bytes
    %elt_size = arith.constant 2 : index   // f16 = 2 bytes
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[elt_size] -> (elt_size * 16)>()[%elt_size]

    // Get MFMA A indexing pattern: returns (row, col) for this lane
    %idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair

    // Apply XOR swizzle to avoid bank conflicts
    %swizzled_idx = func.call @xor_swizzled_mfma_index_16xf16(%idx) : (!index_pair) -> !index_pair
    %swizzled_row, %swizzled_col = aster_utils.struct_extract %swizzled_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte address in LDS: address = m_pos=0, n_pos=0 + swizzled position
    %desc = aster_utils.struct_create(%c0, %c0, %swizzled_row, %swizzled_col, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_vgpr = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    %off_i32 = lsir.from_reg %off_vgpr : !v -> i32
    %byte_address = arith.index_cast %off_i32 : i32 to index

    // Compute banks for an 8-byte (dwordx2) access starting at byte_address
    %transfer_size = arith.constant 8 : index  // dwordx2
    %b0, %b1, %b2, %b3, %b4_, %b5_, %b6_, %b7_ = func.call @lds_banks_for_transfer(%byte_address, %transfer_size)
      : (index, index) -> (index, index, index, index, index, index, index, index)


    // Store 4 banks as dwordx4 at tid position
    %tid = gpu.thread_id x
    %c16 = arith.constant 16 : index
    %b0_i32 = arith.index_cast %b0 : index to i32
    %b1_i32 = arith.index_cast %b1 : index to i32
    %b2_i32 = arith.index_cast %b2 : index to i32
    %b3_i32 = arith.index_cast %b3 : index to i32
    %b0_v = lsir.to_reg %b0_i32 : i32 -> !v
    %b1_v = lsir.to_reg %b1_i32 : i32 -> !v
    %b2_v = lsir.to_reg %b2_i32 : i32 -> !v
    %b3_v = lsir.to_reg %b3_i32 : i32 -> !v
    %data = amdgcn.make_register_range %b0_v, %b1_v, %b2_v, %b3_v : !v, !v, !v, !v
    func.call @store_to_global_dwordx4_wait(%data, %out_ptr, %c0, %tid, %c16)
      : (!vx4, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
