!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @test_indexing target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // indexing.mlir
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @mfma_index_A_16x16xf16() -> (index, index)
  func.func private @swizzled_mfma_index_A_16x16xf16(index) -> (index, index)
  func.func private @lds_banks_for_transfer(index, index) -> (index, index, index, index)
  // copies.mlir
  func.func private @store_to_global_dwordx2_wait(!vx2, !sx2, index, index, index)

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
    %row, %col = func.call @mfma_index_A_16x16xf16() : () -> (index, index)

    // Compute byte address in LDS directly (no swizzle)
    %off_vgpr = func.call @tiled_matrix_offset(
        %c0, %c0, %row, %col, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    %off_i32 = lsir.from_reg %off_vgpr : !v -> i32
    %byte_address = arith.index_cast %off_i32 : i32 to index

    // Compute banks for an 8-byte access
    %transfer_size = arith.constant 8 : index  // dwordx2
    %b0, %b1, %b2, %b3 = func.call @lds_banks_for_transfer(%byte_address, %transfer_size)
      : (index, index) -> (index, index, index, index)


    // Store bank results as dwordx2 at tid position
    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %b0_i32 = arith.index_cast %b0 : index to i32
    %b1_i32 = arith.index_cast %b1 : index to i32
    %b0_v = lsir.to_reg %b0_i32 : i32 -> !v
    %b1_v = lsir.to_reg %b1_i32 : i32 -> !v
    %data = amdgcn.make_register_range %b0_v, %b1_v : !v, !v
    func.call @store_to_global_dwordx2_wait(%data, %out_ptr, %c0, %tid, %c8)
      : (!vx2, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test LDS banks for swizzled MFMA A matrix pattern (16x16xf16).
  // Each thread outputs 2 bank indices (b0, b1) for its swizzled address.
  // Output layout: tid * 8 bytes -> [b0, b1] (2 x i32)
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

    // Apply swizzle pattern to avoid bank conflicts (transfer_size = 8 for typical access)
    %transfer_size = arith.constant 8 : index  // dwordx2
    %swizzled_row, %swizzled_col = func.call @swizzled_mfma_index_A_16x16xf16(%transfer_size)
      : (index) -> (index, index)

    // Compute byte address in LDS: address = m_pos=0, n_pos=0 + swizzled position
    %off_vgpr = func.call @tiled_matrix_offset(
        %c0, %c0, %swizzled_row, %swizzled_col, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    %off_i32 = lsir.from_reg %off_vgpr : !v -> i32
    %byte_address = arith.index_cast %off_i32 : i32 to index

    // Compute banks for an 8-byte (dwordx2) access starting at byte_address
    %b0, %b1, %b2, %b3 = func.call @lds_banks_for_transfer(%byte_address, %transfer_size)
      : (index, index) -> (index, index, index, index)


    // Store bank results as dwordx2 at tid position
    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %b0_i32 = arith.index_cast %b0 : index to i32
    %b1_i32 = arith.index_cast %b1 : index to i32
    %b0_v = lsir.to_reg %b0_i32 : i32 -> !v
    %b1_v = lsir.to_reg %b1_i32 : i32 -> !v
    %data = amdgcn.make_register_range %b0_v, %b1_v : !v, !v
    func.call @store_to_global_dwordx2_wait(%data, %out_ptr, %c0, %tid, %c8)
      : (!vx2, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
