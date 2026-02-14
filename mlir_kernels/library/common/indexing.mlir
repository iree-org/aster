// Common indexing functions for AMDGCN kernels.
// These functions compute byte offsets for tiled data access patterns.

// From descriptors.mlir
!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx4 = !amdgcn.sgpr<[? + 4]>
!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!a   = !amdgcn.agpr
!ax1 = !amdgcn.agpr
!ax2 = !amdgcn.agpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>
!index_descriptor_3level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, iii: index, jjj: index, stride: index, elt_size_b: index>
!index_tuple_8 = !aster_utils.struct<b0: index, b1: index, b2: index, b3: index, b4: index, b5: index, b6: index, b7: index>

amdgcn.library @common_indexing {
  //===--------------------------------------------------------------------===//
  // GPU id functions.
  //===--------------------------------------------------------------------===//
  // Get the lane id within the wave (0..63)
  func.func private @lane_id() -> index {
    %tid = gpu.thread_id x
    %lane_id = affine.apply affine_map<()[tid] -> (tid mod 64)>()[%tid]
    return %lane_id : index
  }

  // Get the wave id within the block.
  // Note: assumes 1-D thread specification ([num_threads, 1, 1]).
  func.func private @wave_id() -> index {
    %tid = gpu.thread_id x
    %wave_id = affine.apply affine_map<()[tid] -> (tid floordiv 64)>()[%tid]
    return %wave_id : index
  }

  // Get the number of waves in the block.
  // Note: assumes 1-D thread specification ([num_threads, 1, 1]).
  func.func private @wave_count() -> index {
    %bdim = gpu.block_dim x
    %wave_count = affine.apply affine_map<()[bdim] -> (bdim ceildiv 64)>()[%bdim]
    return %wave_count : index
  }

  //===--------------------------------------------------------------------===//
  // Reusable work distribution functions.
  //===--------------------------------------------------------------------===//
  // 2-D delinearization of lane id to 2D position.
  func.func private @lane_delinearize_2d(%dims: !index_pair) -> !index_pair {
    %M, %N = aster_utils.struct_extract %dims ["i", "j"] : !index_pair -> index, index
    %lane_id = func.call @lane_id() : () -> index
    %i, %j = affine.delinearize_index %lane_id into (%M, %N) : index, index
    %result = aster_utils.struct_create(%i, %j) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // Compute 2D partitioning of blocks within the grid.
  func.func private @block_id_x_delinearize_2d(%dims: !index_pair) -> !index_pair {
    %M, %N = aster_utils.struct_extract %dims ["i", "j"] : !index_pair -> index, index
    %bid = gpu.block_id x
    %i, %j = affine.delinearize_index %bid into (%M, %N) : index, index
    %result = aster_utils.struct_create(%i, %j) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // Compute 2D partitioning of blocks within the grid for tiled problems.
  func.func private @tiled_grid_partition_2d(
    %sizes: !index_pair,      // Problem sizes (M_SIZE, N_SIZE)
    %tile_sizes: !index_pair  // Tile sizes (M_TILE_SIZE, N_TILE_SIZE)
  ) -> !index_pair {
    %M_SIZE, %N_SIZE = aster_utils.struct_extract %sizes ["i", "j"] : !index_pair -> index, index
    %M_TILE_SIZE, %N_TILE_SIZE = aster_utils.struct_extract %tile_sizes ["i", "j"] : !index_pair -> index, index
    %M = affine.apply affine_map<()[sz, bsz] -> (sz ceildiv bsz)>()[%M_SIZE, %M_TILE_SIZE]
    %N = affine.apply affine_map<()[sz, bsz] -> (sz ceildiv bsz)>()[%N_SIZE, %N_TILE_SIZE]
    %dims = aster_utils.struct_create(%M, %N) : (index, index) -> !index_pair
    %result = func.call @block_id_x_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  //===--------------------------------------------------------------------===//
  // Reusable contiguous memory access indexing functions.
  //===--------------------------------------------------------------------===//

  // Compute the base distributed index for this thread in a 1D grid.
  // Formula: blockidx * blockdim + threadidx
  // This gives each thread a unique index within the entire grid.
  func.func private @distributed_index_1d() -> index {
    %blockidx_x = gpu.block_id x
    %threadidx_x = gpu.thread_id x
    %blockdim_x = gpu.block_dim x
    %base_index = affine.apply affine_map<
      (bidx, tidx)[bdim] -> (bidx * bdim + tidx)>
      (%blockidx_x, %threadidx_x)[%blockdim_x]
    return %base_index : index
  }

  // Compute the grid stride for a 1D grid-stride loop pattern.
  // Formula: griddim * blockdim (total number of threads in the grid)
  func.func private @grid_stride_1d() -> index {
    %blockdim_x = gpu.block_dim x
    %griddim_x = gpu.grid_dim x
    %stride = affine.apply affine_map<
      ()[gdim, bdim] -> (gdim * bdim)>
      ()[%griddim_x, %blockdim_x]
    return %stride : index
  }

  // Compute the linear byte offset for accessing a 2-D matrix given the outer
  // and inner positions.
  func.func private @matrix_offset(%desc: !index_descriptor_2d) -> !v {
    %i, %j, %stride, %elt_size = aster_utils.struct_extract %desc ["i", "j", "stride", "elt_size_b"] : !index_descriptor_2d -> index, index, index, index
    %off = affine.apply
      affine_map<()[i, j, stride, elt_size] -> (i * stride  + j * elt_size)>
      ()[%i, %j, %stride, %elt_size]
    %off_i32 = arith.index_cast %off : index to i32
    %off_reg = lsir.to_reg %off_i32 : i32 -> !v
    return %off_reg : !v
  }

  // Compute the linear byte offset for accessing a tiled 2-D matrix given the
  // positions to the start of the tile and the position within the tile.
  func.func private @tiled_matrix_offset(
    %desc: !index_descriptor_2level_2d
  ) -> !v {
    %i, %j, %ii, %jj, %stride, %elt_size = aster_utils.struct_extract %desc ["i", "j", "ii", "jj", "stride", "elt_size_b"] : !index_descriptor_2level_2d -> index, index, index, index, index, index
    %i_pos = affine.apply affine_map<()[i, ii] -> (i + ii)>()[%i, %ii]
    %j_pos = affine.apply affine_map<()[j, jj] -> (j + jj)>()[%j, %jj]
    %desc_2d = aster_utils.struct_create(%i_pos, %j_pos, %stride, %elt_size) : (index, index, index, index) -> !index_descriptor_2d
    %off_reg = func.call @matrix_offset(%desc_2d) : (!index_descriptor_2d) -> !v
    return %off_reg : !v
  }

  // Compute the linear byte offset for accessing a twice tiled 2-D matrix given the
  // positions to the start of the major tile, positions to the start of the
  // minor tile, and the position within the tile.
  func.func private @tiledx2_matrix_offset(
    %desc: !index_descriptor_3level_2d
  ) -> !v {
    %i, %j, %ii, %jj, %iii, %jjj, %stride, %elt_size = aster_utils.struct_extract %desc ["i", "j", "ii", "jj", "iii", "jjj", "stride", "elt_size_b"] : !index_descriptor_3level_2d -> index, index, index, index, index, index, index, index
    %i_pos = affine.apply affine_map<()[i, ii, iii] -> (i + ii + iii)>()[%i, %ii, %iii]
    %j_pos = affine.apply affine_map<()[j, jj, jjj] -> (j + jj + jjj)>()[%j, %jj, %jjj]
    %desc_2d = aster_utils.struct_create(%i_pos, %j_pos, %stride, %elt_size) : (index, index, index, index) -> !index_descriptor_2d
    %off_reg = func.call @matrix_offset(%desc_2d) : (!index_descriptor_2d) -> !v
    return %off_reg : !v
  }

  //===--------------------------------------------------------------------===//
  // Reusable MFMA memory access indexing functions.
  //===--------------------------------------------------------------------===//
  // MFMA indexing function for accessing the `A` 16x16xf16 fragment
  func.func private @mfma_index_A_16x16xf16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %result = aster_utils.struct_create(%row, %col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `B` 16x16xf16 fragment
  func.func private @mfma_index_B_16x16xf16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %result = aster_utils.struct_create(%col, %row) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `C` 16x16xf32 fragment
  func.func private @mfma_index_C_16x16xf32() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %result = aster_utils.struct_create(%row, %col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  //===--------------------------------------------------------------------===//
  // Reusable swizzles MFMA memory access indexing functions.
  //===--------------------------------------------------------------------===//
  // XOR swizzle for 16-column f16 layout (avoids bank conflicts)
  // Input: (row, col) in fragment coordinates
  // Output: (row, swizzled_col) for LDS access
  // Formula: swizzled_col = col XOR (row / 4)
  // We XOR the high 2 bits of col (col / 4) with row_group (row / 4)
  func.func private @xor_swizzled_mfma_index_16xf16(%idx: !index_pair) -> !index_pair {
    %row, %col = aster_utils.struct_extract %idx ["i", "j"] : !index_pair -> index, index
    // row_group = row / 4 (values 0, 1, 2, 3 for rows 0, 4, 8, 12)
    %row_group = affine.apply affine_map<()[row] -> (row floordiv 4)>()[%row]

    // col_low = col mod 4, col_high = col / 4
    %col_low = affine.apply affine_map<()[col] -> (col mod 4)>()[%col]
    %col_high = affine.apply affine_map<()[col] -> (col floordiv 4)>()[%col]

    // XOR col_high with row_group using arith.xori
    %col_high_i32 = arith.index_cast %col_high : index to i32
    %row_group_i32 = arith.index_cast %row_group : index to i32
    %xored_i32 = arith.xori %col_high_i32, %row_group_i32 : i32
    %xored = arith.index_cast %xored_i32 : i32 to index

    // Reconstruct: swizzled_col = xored * 4 + col_low
    %swizzled_col = affine.apply affine_map<()[xored, col_low]
      -> (xored * 4 + col_low)>()[%xored, %col_low]

    %result = aster_utils.struct_create(%row, %swizzled_col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // Swizzle for `A` 16x16xf16 fragment with bank conflict avoidance
  // A matrix is accessed with transposed pattern (col-major in LDS)
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_A_16x16xf16() -> !index_pair {
    %idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %result = func.call @xor_swizzled_mfma_index_16xf16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  // Swizzle for `B` 16x16xf16 fragment with bank conflict avoidance
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_B_16x16xf16() -> !index_pair {
    %idx = func.call @mfma_index_B_16x16xf16() : () -> !index_pair
    %result = func.call @xor_swizzled_mfma_index_16xf16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  // Swizzle for `C` 16x16xf32 fragment with bank conflict avoidance
  // For f32: each element is 4 bytes = 1 bank width
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_C_16x16xf32() -> !index_pair {
    %idx = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
    %result = func.call @xor_swizzled_mfma_index_16xf16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  //===--------------------------------------------------------------------===//
  // LDS bank computation functions for debugging bank conflicts.
  // AMD GPUs have 32 banks with 2 bytes per bank (64-byte bank cycle).
  // For a byte at address A: bank = (A / 2) % 32
  //===--------------------------------------------------------------------===//

  // Compute bank indices for a contiguous transfer starting at byte_address.
  // AMD LDS has 32 banks with 2 bytes per bank, so bank = (addr / 2) % 32.
  //
  // Args:
  //   %byte_address: Starting byte address in LDS
  //   %transfer_size: Transfer size in bytes (4=b32, 8=b64, 12=b96, 16=b128)
  //
  // Returns 8 bank indices (b0..b7):
  //   -  b32  (4 bytes): b0, b1 valid; b2..b7 = -1
  //   -  b64  (8 bytes): b0..b3 valid; b4..b7 = -1
  //   -  b96 (12 bytes): b0..b5 valid; b6..b7 = -1
  //   - b128 (16 bytes): b0..b7 all valid
  //
  // Traps with code 42 for unsupported transfer sizes.
  func.func private @lds_banks_for_transfer(
    %addr: index,
    %transfer_size: index
  ) -> !index_tuple_8 {
    %neg1 = arith.constant -1 : index

    // Compute all 8 possible banks (this is a thread-local quantity)
    %aaddr = aster_utils.assume_range %addr min 0 : index
    %b0_val = affine.apply affine_map<()[addr] -> (((addr + 0) floordiv 2) mod 32)>()[%aaddr]
    %b1_val = affine.apply affine_map<()[addr] -> (((addr + 2) floordiv 2) mod 32)>()[%aaddr]
    %b2_val = affine.apply affine_map<()[addr] -> (((addr + 4) floordiv 2) mod 32)>()[%aaddr]
    %b3_val = affine.apply affine_map<()[addr] -> (((addr + 6) floordiv 2) mod 32)>()[%aaddr]
    %b4_val = affine.apply affine_map<()[addr] -> (((addr + 8) floordiv 2) mod 32)>()[%aaddr]
    %b5_val = affine.apply affine_map<()[addr] -> (((addr + 10) floordiv 2) mod 32)>()[%aaddr]
    %b6_val = affine.apply affine_map<()[addr] -> (((addr + 12) floordiv 2) mod 32)>()[%aaddr]
    %b7_val = affine.apply affine_map<()[addr] -> (((addr + 14) floordiv 2) mod 32)>()[%aaddr]

    %result = scf.index_switch %transfer_size -> !index_tuple_8
    case 4 {
      // b32: 2 banks valid
      %result_case4 = aster_utils.struct_create(%b0_val, %b1_val, %neg1, %neg1, %neg1, %neg1, %neg1, %neg1) : (index, index, index, index, index, index, index, index) -> !index_tuple_8
      scf.yield %result_case4 : !index_tuple_8
    }
    case 8 {
      // b64: 4 banks valid
      %result_case8 = aster_utils.struct_create(%b0_val, %b1_val, %b2_val, %b3_val, %neg1, %neg1, %neg1, %neg1) : (index, index, index, index, index, index, index, index) -> !index_tuple_8
      scf.yield %result_case8 : !index_tuple_8
    }
    case 12 {
      // b96: 6 banks valid
      %result_case12 = aster_utils.struct_create(%b0_val, %b1_val, %b2_val, %b3_val, %b4_val, %b5_val, %neg1, %neg1) : (index, index, index, index, index, index, index, index) -> !index_tuple_8
      scf.yield %result_case12 : !index_tuple_8
    }
    case 16 {
      // b128: 8 banks valid
      %result_case16 = aster_utils.struct_create(%b0_val, %b1_val, %b2_val, %b3_val, %b4_val, %b5_val, %b6_val, %b7_val) : (index, index, index, index, index, index, index, index) -> !index_tuple_8
      scf.yield %result_case16 : !index_tuple_8
    }
    default {
      %result_default = aster_utils.struct_create(%neg1, %neg1, %neg1, %neg1, %neg1, %neg1, %neg1, %neg1) : (index, index, index, index, index, index, index, index) -> !index_tuple_8
      scf.yield %result_default : !index_tuple_8
    }

    return %result : !index_tuple_8
  }

  //===--------------------------------------------------------------------===//
  // One-off batch mfma indexing function.
  //===--------------------------------------------------------------------===//
  // Compute the linear byte offset for MFMA-style tiled memory access.
  // TODO: find a better name for this function.
  func.func private @index_bxmxnxk_16x16x16_f16f16f32(
    %bidx: index, %tidx: index,
    %i: index, %j: index,
    %szI: index, %szJ: index,
    %bdimx: index,
    %tile_size: index,
    %lane_stride: index
  ) -> index {
    %num_waves = affine.apply affine_map<()[bdimx] -> (bdimx floordiv 64)>()[%bdimx]
    %widx = affine.apply affine_map<()[tidx] -> (tidx floordiv 64)>()[%tidx]
    %lidx = affine.apply affine_map<()[tidx] -> (tidx mod 64)>()[%tidx]
    %offset = affine.apply affine_map<
      (bidx, widx, i, j, lidx)[num_waves, szI, szJ, tile_sz, lane_stride]
        -> (bidx * num_waves * szI * szJ * tile_sz +
                        widx * szI * szJ * tile_sz +
                                 i * szJ * tile_sz +
                                       j * tile_sz +
                                       lidx * lane_stride)>
      (%bidx, %widx, %i, %j, %lidx)[%num_waves, %szI, %szJ, %tile_size, %lane_stride]

    return %offset : index
  }
}
