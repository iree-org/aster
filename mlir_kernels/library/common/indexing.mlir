// Common indexing functions for AMDGCN kernels.
// These functions compute byte offsets for tiled data access patterns.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

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

  // Get the wave id within the block
  func.func private @wave_id() -> index {
    %tid = gpu.thread_id x
    %wave_id = affine.apply affine_map<()[tid] -> (tid floordiv 64)>()[%tid]
    return %wave_id : index
  }

  // Get the number of waves in the block
  func.func private @wave_count() -> index {
    %bdim = gpu.block_dim x
    %wave_count = affine.apply affine_map<()[bdim] -> (bdim ceildiv 64)>()[%bdim]
    return %wave_count : index
  }

  //===--------------------------------------------------------------------===//
  // Reusable work distribution functions.
  //===--------------------------------------------------------------------===//
  // 2-D delinearization of lane id to 2D position.
  func.func private @lane_delinearize_2d(%M: index, %N: index) -> (index, index) {
    %lane_id = func.call @lane_id() : () -> index
    %i, %j = affine.delinearize_index %lane_id into (%M, %N) : index, index
    return %i, %j : index, index
  }

  // Compute 2D partitioning of blocks within the grid.
  func.func private @block_id_x_delinearize_2d(%M: index, %N: index) -> (index, index) {
    %bid = gpu.block_id x
    %i, %j = affine.delinearize_index %bid into (%M, %N) : index, index
    return %i, %j : index, index
  }

  // Compute 2D partitioning of blocks within the grid for tiled problems.
  func.func private @tiled_grid_partition_2D(
    %M_SIZE: index,      // Outer problem size
    %N_SIZE: index,      // Inner problem size
    %M_TILE_SIZE: index, // Outer tile size
    %N_TILE_SIZE: index  // Inner tile size
  ) -> (index, index) {
    %M = affine.apply affine_map<()[sz, bsz] -> (sz ceildiv bsz)>()[%M_SIZE, %M_TILE_SIZE]
    %N = affine.apply affine_map<()[sz, bsz] -> (sz ceildiv bsz)>()[%N_SIZE, %N_TILE_SIZE]
    %i, %j = func.call @block_id_x_delinearize_2d(%M, %N) : (index, index) -> (index, index)
    return %i, %j : index, index
  }

  //===--------------------------------------------------------------------===//
  // Reusable contiguous memory access indexing functions.
  //===--------------------------------------------------------------------===//
  // Compute the linear byte offset for accessing a 2-D matrix given the outer
  // and inner positions.
  func.func private @matrix_offset(
    %i: index,       // The outer-most position (e.g. a row)
    %j: index,       // The inner-most position (e.g. a column)
    %stride: index,  // The stride (e.g. inner 2-D size **in bytes**)
    %elt_size: index // The element size **in bytes**
  ) -> !v {
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
    %i: index,       // The outer-most tile position (e.g. the start row of a tile)
    %j: index,       // The inner-most tile position (e.g. the start column of a tile)
    %ii: index,      // The outer-most position (e.g. a row relative to the tile)
    %jj: index,      // The inner-most position (e.g. a column relative to the tile)
    %stride: index,  // The stride (e.g. inner 2-D tile size **in bytes**)
    %elt_size: index // The element size **in bytes**
  ) -> !v {
    %i_pos = affine.apply affine_map<()[i, ii] -> (i + ii)>()[%i, %ii]
    %j_pos = affine.apply affine_map<()[j, jj] -> (j + jj)>()[%j, %jj]
    %off_reg = func.call @matrix_offset(%i_pos, %j_pos, %stride, %elt_size)
      : (index, index, index, index) -> !v
    return %off_reg : !v
  }

  // Compute the linear byte offset for accessing a twice tiled 2-D matrix given the
  // positions to the start of the major tile, positions to the start of the
  // minor tile, and the position within the tile.
  func.func private @tiledx2_matrix_offset(
    %i: index,       // The outer-most major tile position (e.g. the start row of a tile)
    %j: index,       // The inner-most major tile position (e.g. the start column of a tile)
    %ii: index,      // The outer-most minor tile position (e.g. the start row of a the sub-tile)
    %jj: index,      // The inner-most minor tile position (e.g. the start column of the sub-tile)
    %iii: index,     // The outer-most position (e.g. a row relative to the sub-tile)
    %jjj: index,     // The inner-most position (e.g. a column relative to the sub-tile)
    %stride: index,  // The stride (e.g. inner 2-D tile size **in bytes**)
    %elt_size: index // The element size **in bytes**
  ) -> !v {
    %i_pos = affine.apply affine_map<()[i, ii, iii] -> (i + ii + iii)>()[%i, %ii, %iii]
    %j_pos = affine.apply affine_map<()[j, jj, jjj] -> (j + jj + jjj)>()[%j, %jj, %jjj]
    %off_reg = func.call @matrix_offset(%i_pos, %j_pos, %stride, %elt_size)
      : (index, index, index, index) -> !v
    return %off_reg : !v
  }


  //===--------------------------------------------------------------------===//
  // Reusable swizzled memory access indexing functions.
  //===--------------------------------------------------------------------===//
  // Helper to compute swizzled positions for a 16x16 fragment
  // Returns (4 * (lane_id / 16), lane_id mod 16)
  func.func private @mfma_index_16x16_helper() -> (index, index) {
    %lane_id = func.call @lane_id() : () -> index
    %idx0 = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %idx1 = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    return %idx0, %idx1 : index, index
  }

  // Swizzle function for accessing the `A` 16x16xf16 fragment
  func.func private @mfma_index_A_16x16xf16() -> (index, index) {
    // Note the swapped return order
    %j, %i = func.call @mfma_index_16x16_helper() : () -> (index, index)
    return %i, %j : index, index
  }

  // Swizzle function for accessing the `B` 16x16xf16 fragment
  func.func private @mfma_index_B_16x16xf16() -> (index, index) {
    %i, %j = func.call @mfma_index_16x16_helper() : () -> (index, index)
    return %i, %j : index, index
  }

  // Swizzle function for accessing the `C` 16x16xf32 fragment
  func.func private @mfma_index_C_16x16xf32() -> (index, index) {
    %i, %j = func.call @mfma_index_16x16_helper() : () -> (index, index)
    return %i, %j : index, index
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
