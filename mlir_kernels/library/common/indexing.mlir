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
  // Linearized thread ID: tx + bdx * (ty + bdy * tz).
  func.func private @linear_thread_id() -> index {
    %tx = gpu.thread_id x
    %ty = gpu.thread_id y
    %tz = gpu.thread_id z
    %bdx = gpu.block_dim x
    %bdy = gpu.block_dim y
    %ltid = affine.apply affine_map<(tx, ty, tz)[bdx, bdy] -> (tx + bdx * (ty + bdy * tz))>
        (%tx, %ty, %tz)[%bdx, %bdy]
    return %ltid : index
  }

  // Linearized block ID: bx + gdx * (by + gdy * bz).
  func.func private @linear_block_id() -> index {
    %bx = gpu.block_id x
    %by = gpu.block_id y
    %bz = gpu.block_id z
    %gdx = gpu.grid_dim x
    %gdy = gpu.grid_dim y
    %lbid = affine.apply affine_map<(bx, by, bz)[gdx, gdy] -> (bx + gdx * (by + gdy * bz))>
        (%bx, %by, %bz)[%gdx, %gdy]
    return %lbid : index
  }

  // Lane ID within the wave: linear_thread_id mod 64.
  func.func private @lane_id() -> index {
    %ltid = func.call @linear_thread_id() : () -> index
    %lane_id = affine.apply affine_map<()[tid] -> (tid mod 64)>()[%ltid]
    return %lane_id : index
  }

  // Wave ID within the block: linear_thread_id floordiv 64.
  // Consistent with KernelBuilder.wave_id().
  func.func private @wave_id() -> index {
    %ltid = func.call @linear_thread_id() : () -> index
    %wave_id = affine.apply affine_map<()[tid] -> (tid floordiv 64)>()[%ltid]
    return %wave_id : index
  }

  // Number of waves in the block.
  func.func private @wave_count() -> index {
    %bdx = gpu.block_dim x
    %bdy = gpu.block_dim y
    %bdz = gpu.block_dim z
    %total = affine.apply affine_map<()[x, y, z] -> (x * y * z)>()[%bdx, %bdy, %bdz]
    %wave_count = affine.apply affine_map<()[t] -> (t ceildiv 64)>()[%total]
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
  // MFMA indexing function for accessing the `A` 16x16_f16 fragment
  func.func private @mfma_index_A_16x16_f16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %result = aster_utils.struct_create(%row, %col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `B` 16x16_f16 fragment
  func.func private @mfma_index_B_16x16_f16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 4))>()[%lane_id]
    %result = aster_utils.struct_create(%col, %row) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `A` 16x32xfp8 fragment
  // FP8 16x16x32: Lane L loads row = L % 16, col = (L / 16) * 8
  // Each lane holds 8 consecutive fp8 elements (8 bytes = dwordx2)
  func.func private @mfma_index_A_16x16xfp8() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 8))>()[%lane_id]
    %result = aster_utils.struct_create(%row, %col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `B` 32x16xfp8 fragment
  // Same physical layout as A; MFMA handles the transpose internally
  func.func private @mfma_index_B_16x16xfp8() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 16)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> (((lid floordiv 16) * 8))>()[%lane_id]
    %result = aster_utils.struct_create(%col, %row) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing function for accessing the `C` 16x16_f32 fragment
  func.func private @mfma_index_C_16x16_f32() -> !index_pair {
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
  func.func private @swizzled_mfma_index_16_f16(%idx: !index_pair) -> !index_pair {
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

  // Swizzle for `A` 16x16_f16 fragment with bank conflict avoidance
  // A matrix is accessed with transposed pattern (col-major in LDS)
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_A_16x16_f16() -> !index_pair {
    %idx = func.call @mfma_index_A_16x16_f16() : () -> !index_pair
    %result = func.call @swizzled_mfma_index_16_f16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  // Swizzle for `B` 16x16_f16 fragment with bank conflict avoidance
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_B_16x16_f16() -> !index_pair {
    %idx = func.call @mfma_index_B_16x16_f16() : () -> !index_pair
    %result = func.call @swizzled_mfma_index_16_f16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  // Swizzle for `C` 16x16_f32 fragment with bank conflict avoidance
  // For f32: each element is 4 bytes = 1 bank width
  // Returns (row, swizzled_col) for LDS access
  func.func private @swizzled_mfma_index_C_16x16_f32() -> !index_pair {
    %idx = func.call @mfma_index_C_16x16_f32() : () -> !index_pair
    %result = func.call @swizzled_mfma_index_16_f16(%idx) : (!index_pair) -> !index_pair
    return %result : !index_pair
  }

  //===--------------------------------------------------------------------===//
  // 32x32x8 MFMA indexing functions (CDNA3/4)
  //===--------------------------------------------------------------------===//

  // MFMA indexing for A 32x32x8 f16 fragment.
  // Lane l holds row l%32 of A, cols [(l/32)*4, (l/32)*4+3].
  // Returns (row, col_start).
  func.func private @mfma_index_A_32x32_f16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 32)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> ((lid floordiv 32) * 4)>()[%lane_id]
    %result = aster_utils.struct_create(%row, %col) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing for B 32x32x8 f16 fragment.
  // Same physical layout as A; returns (col, row) for transposed access.
  func.func private @mfma_index_B_32x32_f16() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid mod 32)>()[%lane_id]
    %col = affine.apply affine_map<()[lid] -> ((lid floordiv 32) * 4)>()[%lane_id]
    %result = aster_utils.struct_create(%col, %row) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // MFMA indexing for C/D 32x32 f32 output fragment.
  // Returns (col_pos, row_base) where:
  //   col_pos = lane_id % 32 (column in output matrix)
  //   row_base = (lane_id / 32) * 4 (0 for lanes 0-31, 4 for lanes 32-63)
  // Full row for register r: row_base + 8*(r/4) + r%4
  func.func private @mfma_index_C_32x32_f32() -> !index_pair {
    %lane_id = func.call @lane_id() : () -> index
    %col = affine.apply affine_map<()[lid] -> (lid mod 32)>()[%lane_id]
    %row_base = affine.apply affine_map<()[lid] -> ((lid floordiv 32) * 4)>()[%lane_id]
    %result = aster_utils.struct_create(%col, %row_base) : (index, index) -> !index_pair
    return %result : !index_pair
  }

  // Compute the row index for register r in a 32x32 MFMA C fragment.
  // Formula: row_base + 8*(r floordiv 4) + r mod 4
  // Where row_base comes from @mfma_index_C_32x32_f32 (j field).
  // 4 groups of 4 consecutive rows, spaced 8 apart:
  //   r=0..3  -> row_base+0..3
  //   r=4..7  -> row_base+8..11
  //   r=8..11 -> row_base+16..19
  //   r=12..15 -> row_base+24..27
  func.func private @mfma_c_row_32x32_f32(%row_base: index, %reg_idx: index) -> index {
    %row = affine.apply affine_map<(rb, r) -> (rb + 8 * (r floordiv 4) + r mod 4)>(%row_base, %reg_idx)
    return %row : index
  }

  // Compute byte offset for a 2D global tile access.
  // Formula: (tile_row + thread_row) * stride + tile_col * elt_size + thread_byte
  // Used when a wavefront fills tiles: (tile_row, tile_col) is the tile origin,
  // (thread_row, thread_byte) is the per-thread delta within the tile.
  func.func private @tiled_row_byte_off(
      %tile_row: index, %thread_row: index, %tile_col: index, %thread_byte: index,
      %stride: index, %elt_size: index
  ) -> index {
    %off = affine.apply
        affine_map<()[tr, dr, tc, db, s, e] -> ((tr + dr) * s + tc * e + db)>
        ()[%tile_row, %thread_row, %tile_col, %thread_byte, %stride, %elt_size]
    return %off : index
  }

  // Byte offset for register i (0..3) of the MFMA 16x16 f32 C fragment.
  // Each thread holds 4 consecutive output rows: row_base+0, row_base+1, row_base+2, row_base+3.
  // row_base and col come from @mfma_index_C_16x16_f32 ({i: col, j: row_base}).
  // Formula: (m + row_base + i) * stride + (n + col) * elt_size
  func.func private @mfma_c_16x16_f32_byte_offset(
      %m: index, %n: index, %row_base: index, %col: index,
      %stride: index, %elt_size: index, %i: index
  ) -> index {
    %off = affine.apply
        affine_map<()[m, n, rb, col, s, e, i] -> ((m + rb + i) * s + (n + col) * e)>
        ()[%m, %n, %row_base, %col, %stride, %elt_size, %i]
    return %off : index
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

  //===--------------------------------------------------------------------===//
  // 32x32 tile cooperative thread mapping and LDS swizzle.
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row_in_group, col) for 32x32 tile cooperative fill.
  // 64 lanes, 8 threads per row, 4 f16 elements per thread:
  //   row_in_group = lane_id / 8 (0..7)
  //   col = (lane_id % 8) * 4 (0, 4, 8, ..., 28)
  // 4 load/store cycles cover all 32 rows (row groups 0-7, 8-15, 16-23, 24-31).
  func.func private @thread_tile_pos_32x32() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 8)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 8) * 4)>()[%lane]
    return %row, %col : index, index
  }

  // Compute XOR-swizzled LDS byte address for 32x32 f16 tiles (stride = 64 bytes/row).
  // Swizzle: base + row*64 + (byte_in_row XOR mask)
  //   mask = ((row / 2) % 8) * 8 -- permutes bits [5:3] based on row.
  // Uses arith.xori since affine_map cannot express XOR.
  // NOTE: different from @swizzled_mfma_index_16_f16 which XORs column-level
  // indices (row/4) for 16-col tiles. This operates on byte addresses with
  // (row/2)%8 for 32-col tiles.
  func.func private @lds_swizzled_addr_32x32(%base: index, %row: index, %byte_in_row: index) -> index {
    %mask = affine.apply affine_map<(r) -> (((r floordiv 2) mod 8) * 8)>(%row)
    %mask_i32 = arith.index_cast %mask : index to i32
    %byte_i32 = arith.index_cast %byte_in_row : index to i32
    %swizzled_i32 = arith.xori %byte_i32, %mask_i32 : i32
    %swizzled = arith.index_cast %swizzled_i32 : i32 to index
    %addr = affine.apply affine_map<()[b, r, s] -> (b + r * 64 + s)>()[%base, %row, %swizzled]
    return %addr : index
  }

  //===--------------------------------------------------------------------===//
  // 16x32 tile cooperative thread mapping and LDS swizzle (dwordx4, 16x16 f16 sub-tiles).
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row, col) for cooperative fill of a 16x32 f16 tile via dwordx4 loads.
  // 64 lanes, 8 f16 elements per lane:
  //   row = lane_id / 4  (0..15)
  //   col = (lane_id % 4) * 8  (0, 8, 16, 24)
  func.func private @thread_tile_pos_16x32_f16() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 4) * 8)>()[%lane]
    return %row, %col : index, index
  }

  // Compute XOR-swizzled LDS byte offset for a 16-row f16 sub-tile (stride = 32 bytes/row).
  // Sub-tile occupies 16*32 = 512 bytes; two sub-tiles cover a 16x32 tile (K0 at base,
  // K1 at base+512).
  // Swizzle formula: swizzled_col = (col/4 XOR row/4)*4 + col%4
  //   offset = tile_base + row*32 + swizzled_col*2
  // Uses arith.xori since affine_map cannot express XOR.
  func.func private @lds_swizzled_byte_offset_16row_f16(
      %tile_base: index, %row: index, %col: index
  ) -> index {
    %row_group = affine.apply affine_map<()[r] -> (r floordiv 4)>()[%row]
    %col_low = affine.apply affine_map<()[c] -> (c mod 4)>()[%col]
    %col_high = affine.apply affine_map<()[c] -> (c floordiv 4)>()[%col]

    %col_high_i32 = arith.index_cast %col_high : index to i32
    %row_group_i32 = arith.index_cast %row_group : index to i32
    %xored_i32 = arith.xori %col_high_i32, %row_group_i32 : i32
    %xored = arith.index_cast %xored_i32 : i32 to index

    %swizzled_col = affine.apply affine_map<()[x, cl] -> (x * 4 + cl)>()[%xored, %col_low]
    %offset = affine.apply affine_map<()[base, r, sc] -> (base + r * 32 + sc * 2)>()
        [%tile_base, %row, %swizzled_col]
    return %offset : index
  }

  //===--------------------------------------------------------------------===//
  // 16x64_b tile cooperative thread mapping and LDS swizzle (dwordx4, full 16x32 f16 in one tile).
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row, col_byte) for cooperative fill of a 16x32 f16 tile via dwordx4 loads.
  // 64 lanes, 8 f16 elements per lane (16 bytes per lane):
  //   row = lane_id / 4  (0..15)
  //   col_byte = (lane_id % 4) * 16  (0, 16, 32, 48)
  // Each thread covers 16 consecutive bytes in its row: col_byte..col_byte+15.
  // Two ds_write_b64 per thread: lo at col_byte, hi at col_byte+8.
  func.func private @thread_tile_pos_16x64_b() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col_byte = affine.apply affine_map<()[lid] -> ((lid mod 4) * 16)>()[%lane]
    return %row, %col_byte : index, index
  }

  // Compute ds_bpermute byte address to rearrange a 16x64_b cooperative fill
  // into an MFMA A fragment for v_mfma_f32_16x16x16_f16.
  //
  // Cooperative fill: lane i holds row=i/4, cols [(i%4)*8..(i%4)*8+7] (f16 elts).
  // MFMA A fragment:  lane d needs row=d%16, cols [(d/16)*4..(d/16)*4+3].
  // Source lane: src(d, k_sel) = 4*(d%16) + (d/32) + k_sel*2
  // Bpermute addr (bytes): 4 * src = 16*(d%16) + 4*(d/32) + 8*k_sel
  //
  // k_sel: 0 for K0 (columns 0-15), 1 for K1 (columns 16-31).
  func.func private @bpermute_addr_A_16x16_f16(%k_sel: index) -> index {
    %lane = func.call @lane_id() : () -> index
    %addr = affine.apply affine_map<(lid)[ks] -> (
        (lid mod 16) * 16 + (lid floordiv 32) * 4 + ks * 8
    )>(%lane)[%k_sel]
    return %addr : index
  }

  // Compute the VGPR-pair select mask for A fragment extraction from 16x64_b.
  //
  // After ds_bpermute of all 4 source VGPRs, we need to select the correct pair:
  //   vbase = ((lane_id/16) % 2) -- 0 for lanes [0-15,32-47], 1 for [16-31,48-63]
  //
  // Returns 0 or 1 as an index. Caller uses v_cmp_eq_i32_e64 to build a lane mask:
  //   mask = (vbase == 0) -> cndmask picks bp[i] (lo pair) or bp[i+2] (hi pair).
  func.func private @vbase_select_A_16x16_f16() -> index {
    %lane = func.call @lane_id() : () -> index
    %vbase = affine.apply affine_map<(lid) -> ((lid floordiv 16) mod 2)>(%lane)
    return %vbase : index
  }

  // Compute XOR-swizzled LDS byte address for a 16-row tile with 64-byte row stride.
  // Swizzle: base + row*64 + (byte_in_row XOR mask)
  //   mask = ((row / 2) % 8) * 8
  // For 16 rows: mask cycles through 8 distinct values (0,8,16,...,56), each used for 2 rows.
  // This is identical in formula to @lds_swizzled_addr_32x32; named separately for clarity.
  // Uses arith.xori since affine_map cannot express XOR.
  func.func private @lds_swizzled_addr_16x64_b(%base: index, %row: index, %byte_in_row: index) -> index {
    %mask = affine.apply affine_map<(r) -> (((r floordiv 2) mod 8) * 8)>(%row)
    %mask_i32 = arith.index_cast %mask : index to i32
    %byte_i32 = arith.index_cast %byte_in_row : index to i32
    %swizzled_i32 = arith.xori %byte_i32, %mask_i32 : i32
    %swizzled = arith.index_cast %swizzled_i32 : i32 to index
    %addr = affine.apply affine_map<()[b, r, s] -> (b + r * 64 + s)>()[%base, %row, %swizzled]
    return %addr : index
  }
}
