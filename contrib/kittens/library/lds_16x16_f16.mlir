// Kittens LDS (Local Data Share / Shared Memory) primitives for 16x16 f16 tiles.
//
// Allocation, addressing, and transfer functions for moving data between
// global memory, LDS, and registers.
//
// Async by default: load functions return futures/tokens, callers wait explicitly.
// Follows the same convention as global_16x16_f16.mlir (load returns future/token,
// get extracts value).
//
// Supports:
// 1. 16x16 f16 tiles with padded stride (17 columns) to avoid bank conflicts on
// flat / coalesced indexing.
// 2. 16x16 f16 unpadded tiles for XOR swizzle bank conflict avoidance.
//
// Supports manual multi-buffering (1, 2, or 3 buffers) for latency hiding.
//
// Composes with transformations for automating pipelining and multi-buffering to
// any depth.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2

// Future type for async LDS writes (mirrors !future_lds_write from futures.mlir)
!future_lds_write = !amdgcn.write_token<shared>

// Future type for async LDS reads (mirrors !future_global_read from global_16x16_f16.mlir)
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

amdgcn.library @kittens_lds_16x16_f16 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_B_16x16xf16() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @alloc_vgprx2() -> !vx2

  // From futures.mlir
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // LDS Allocation Functions - Padded (544 bytes/tile, 17-column stride)
  //===--------------------------------------------------------------------===//

  // Allocate LDS for 1-buffer padded configuration (baseline, no latency hiding)
  // Returns: (A_buffer_0, B_buffer_0) as LDS allocation handles
  //
  // Layout:
  //   A[0]: size 544 bytes
  //   B[0]: size 544 bytes
  //   Total: 1,088 bytes
  func.func private @alloc_lds_1buffer_padded() -> (index, index) {
    %A_base = amdgcn.alloc_lds 544
    %B_base = amdgcn.alloc_lds 544

    %A_off = amdgcn.get_lds_offset %A_base : index
    %B_off = amdgcn.get_lds_offset %B_base : index

    return %A_off, %B_off : index, index
  }

  // Allocate LDS for 2-buffer padded configuration (double buffering)
  // Returns: (A_buffer_0, B_buffer_0, A_buffer_1, B_buffer_1) as LDS offsets
  //
  // Layout:
  //   A[0]: size 544 bytes
  //   B[0]: size 544 bytes
  //   A[1]: size 544 bytes
  //   B[1]: size 544 bytes
  //   Total: 2,176 bytes
  func.func private @alloc_lds_2buffer_padded() -> (index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 544
    %B0_alloc = amdgcn.alloc_lds 544
    %A1_alloc = amdgcn.alloc_lds 544
    %B1_alloc = amdgcn.alloc_lds 544

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index

    return %A0, %B0, %A1, %B1 : index, index, index, index
  }

  // Allocate LDS for 3-buffer padded configuration (triple buffering)
  // Returns: (A[0], B[0], A[1], B[1], A[2], B[2]) as LDS offsets
  //
  // Layout:
  //   A[0]: size 544 bytes
  //   B[0]: size 544 bytes
  //   A[1]: size 544 bytes
  //   B[1]: size 544 bytes
  //   A[2]: size 544 bytes
  //   B[2]: size 544 bytes
  //   Total: 3,264 bytes
  func.func private @alloc_lds_3buffer_padded() -> (index, index, index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 544
    %B0_alloc = amdgcn.alloc_lds 544
    %A1_alloc = amdgcn.alloc_lds 544
    %B1_alloc = amdgcn.alloc_lds 544
    %A2_alloc = amdgcn.alloc_lds 544
    %B2_alloc = amdgcn.alloc_lds 544

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index
    %A2 = amdgcn.get_lds_offset %A2_alloc : index
    %B2 = amdgcn.get_lds_offset %B2_alloc : index

    return %A0, %B0, %A1, %B1, %A2, %B2 : index, index, index, index, index, index
  }

  //===--------------------------------------------------------------------===//
  // LDS Allocation Functions - XOR Swizzle (512 bytes/tile, default)
  //===--------------------------------------------------------------------===//

  // 1-buffer: 2 tiles x 512 = 1,024 bytes total
  func.func private @alloc_lds_1buffer() -> (index, index) {
    %A_base = amdgcn.alloc_lds 512
    %B_base = amdgcn.alloc_lds 512

    %A_off = amdgcn.get_lds_offset %A_base : index
    %B_off = amdgcn.get_lds_offset %B_base : index

    return %A_off, %B_off : index, index
  }

  // 2-buffer: 4 tiles x 512 = 2,048 bytes total
  func.func private @alloc_lds_2buffer() -> (index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 512
    %B0_alloc = amdgcn.alloc_lds 512
    %A1_alloc = amdgcn.alloc_lds 512
    %B1_alloc = amdgcn.alloc_lds 512

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index

    return %A0, %B0, %A1, %B1 : index, index, index, index
  }

  // 3-buffer: 6 tiles x 512 = 3,072 bytes total
  func.func private @alloc_lds_3buffer() -> (index, index, index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 512
    %B0_alloc = amdgcn.alloc_lds 512
    %A1_alloc = amdgcn.alloc_lds 512
    %B1_alloc = amdgcn.alloc_lds 512
    %A2_alloc = amdgcn.alloc_lds 512
    %B2_alloc = amdgcn.alloc_lds 512

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index
    %A2 = amdgcn.get_lds_offset %A2_alloc : index
    %B2 = amdgcn.get_lds_offset %B2_alloc : index

    return %A0, %B0, %A1, %B1, %A2, %B2 : index, index, index, index, index, index
  }

  //===--------------------------------------------------------------------===//
  // LDS Addressing Functions
  //===--------------------------------------------------------------------===//

  // Padded addressing: stride = 17 columns (34 bytes/row), wastes 6% LDS.
  // offset = tile_base + row * 34 + col * 2
  // Must pair with @alloc_lds_Nbuffer_padded() (544-byte tiles).
  func.func private @lds_element_offset_padded(
      %tile_base: index,
      %row: index,
      %col: index
  ) -> index {
    %offset = affine.apply affine_map<()[base, row, col] -> (base + row * 34 + col * 2)>
        ()[%tile_base, %row, %col]
    return %offset : index
  }

  // XOR swizzle addressing: stride = 16 columns (32 bytes/row), zero overhead.
  // Formula: swizzled_col = (col/4 XOR row/4)*4 + col%4
  // offset = tile_base + row * 32 + swizzled_col * 2
  // Must pair with @alloc_lds_Nbuffer() (512-byte tiles).
  func.func private @lds_element_offset_xor_swizzle(
      %tile_base: index,
      %row: index,
      %col: index
  ) -> index {
    %row_group = affine.apply affine_map<()[r] -> (r floordiv 4)>()[%row]
    %col_low = affine.apply affine_map<()[c] -> (c mod 4)>()[%col]
    %col_high = affine.apply affine_map<()[c] -> (c floordiv 4)>()[%col]

    %col_high_i32 = arith.index_cast %col_high : index to i32
    %row_group_i32 = arith.index_cast %row_group : index to i32
    %xored_i32 = arith.xori %col_high_i32, %row_group_i32 : i32
    %xored = arith.index_cast %xored_i32 : i32 to index

    %swizzled_col = affine.apply affine_map<()[x, cl] -> (x * 4 + cl)>()[%xored, %col_low]
    %offset = affine.apply affine_map<()[base, r, sc] -> (base + r * 32 + sc * 2)>
        ()[%tile_base, %row, %swizzled_col]
    return %offset : index
  }

  //===--------------------------------------------------------------------===//
  // Thread-to-Element Mapping
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row, col) for cooperative LDS loads of a 16x16 tile.
  // 64 lanes, 4 elements/lane, row-major: row = lane floordiv 4, col = (lane mod 4) * 4
  func.func private @thread_lds_slice() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 4) * 4)>()[%lane]
    return %row, %col : index, index
  }

  //===--------------------------------------------------------------------===//
  // Global -> LDS Transfers
  //===--------------------------------------------------------------------===//
  // Notes
  // =====
  // - This function has an internal synchronization on global memory, this is
  // MI300 specific and only useful for testing (it breaks the future contract
  // internally).
  // - On MI350 and later this is the right API surface because the DMA has
  // global to load semantics without using registers. However on MI350 the DMA
  // does not swizzle.
  // - Later architectures may introduce swizzles, transposes etc directly in
  // the DMA, the API will be ready for those.

  // 16x16xf16 global load -> LDS using flat writes to LDS.
  // Issues global load, waits for data in registers, fires LDS write, and
  // returns the ds_write future. The caller must wait on the future (and
  // s_barrier for cross-wave visibility when in multi-wave mode) before reading
  // from LDS.
  func.func private @load_global_to_lds_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !future_lds_write {
    %row, %col = func.call @thread_lds_slice() : () -> (index, index)

    // Compute global memory offset for this thread's slice
    %elt_size = arith.constant 2 : index  // f16 = 2 bytes
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %global_off_vgpr = func.call @tiled_matrix_offset(%desc)
        : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes from global memory
    %c0_i32 = arith.constant 0 : i32
    %tmp_reg = func.call @alloc_vgprx2() : () -> !vx2
    %loaded, %tok_global = amdgcn.load global_load_dwordx2 dest %tmp_reg addr %global_ptr
        offset d(%global_off_vgpr) + c(%c0_i32)
        : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok_global : !amdgcn.read_token<flat>

    // Compute LDS address and write (fire and return future)
    %lds_offset_idx = func.call @lds_element_offset_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0_i32_2 = arith.constant 0 : i32
    %tok_lds = amdgcn.store ds_write_b64 data %loaded addr %lds_addr offset c(%c0_i32_2)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok_lds : !future_lds_write
  }

  // 16x16xf16 global load -> LDS using swizzled write to LDS.
  func.func private @load_global_to_lds_xor_swizzle_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !future_lds_write {
    %row, %col = func.call @thread_lds_slice() : () -> (index, index)

    %elt_size = arith.constant 2 : index
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %global_off_vgpr = func.call @tiled_matrix_offset(%desc)
        : (!index_descriptor_2level_2d) -> !v

    %c0_i32 = arith.constant 0 : i32
    %tmp_reg = func.call @alloc_vgprx2() : () -> !vx2
    %loaded, %tok_global = amdgcn.load global_load_dwordx2 dest %tmp_reg addr %global_ptr
        offset d(%global_off_vgpr) + c(%c0_i32)
        : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok_global : !amdgcn.read_token<flat>

    %lds_offset_idx = func.call @lds_element_offset_xor_swizzle(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0_i32_2 = arith.constant 0 : i32
    %tok_lds = amdgcn.store ds_write_b64 data %loaded addr %lds_addr offset c(%c0_i32_2)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok_lds : !future_lds_write
  }

  //===--------------------------------------------------------------------===//
  // Read from LDS
  //===--------------------------------------------------------------------===//
  // Async by default: load_lds_* issues ds_read and returns a future,
  // get_lds_* waits on the future and returns the tile value.
  // Mirrors the load_A_f16 / get_A_f16 pattern from global_16x16_f16.mlir.

  // A fragment: read from row-major LDS using A indexing.
  // mfma_index_A returns (i=row, j=col) which maps directly to row-major LDS.
  func.func private @load_lds_A_f16(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

  // B fragment: read from row-major LDS using B indexing.
  // mfma_index_B returns (i=col, j=row) -- transposed because MFMA computes
  // A @ B^T. We swap (i,j) -> (row=j, col=i) for row-major LDS access,
  // same as global_16x16_f16.mlir::load_B_f16 uses A indexing for row-major memory.
  func.func private @load_lds_B_f16(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_B_16x16xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

  // A fragment: XOR swizzle addressing -- issue ds_read, return future.
  func.func private @load_lds_A_xor_swizzle_f16(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_xor_swizzle(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

  // B fragment: XOR swizzle addressing with B transposed indexing.
  // mfma_index_B returns (i=col, j=row); swap for row-major LDS, then swizzle.
  func.func private @load_lds_B_xor_swizzle_f16(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_B_16x16xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_xor_swizzle(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

  // Wait on future and extract tile value.
  func.func private @get_lds_A_f16(%future: !future_lds_read) -> !rt_A_f16 {
    %result = func.call @get_lds_read_value_vx2(%future)
        : (!future_lds_read) -> !vx2
    return %result : !rt_A_f16
  }

  func.func private @get_lds_B_f16(%future: !future_lds_read) -> !rt_B_f16 {
    %result = func.call @get_lds_read_value_vx2(%future)
        : (!future_lds_read) -> !vx2
    return %result : !rt_B_f16
  }

  //===--------------------------------------------------------------------===//
  // Write to LDS
  //===--------------------------------------------------------------------===//
  // Async by default: returns ds_write future, caller waits.

  // A fragment: write to row-major LDS using A indexing.
  func.func private @store_lds_A_f16(%tile: !rt_A_f16, %lds_tile_base: index) -> !future_lds_write {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store ds_write_b64 data %tile addr %lds_addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok : !future_lds_write
  }

  // B fragment: write to row-major LDS using B transposed indexing.
  // mfma_index_B returns (i=col, j=row); swap for row-major LDS access.
  func.func private @store_lds_B_f16(%tile: !rt_B_f16, %lds_tile_base: index) -> !future_lds_write {
    %mfma_idx = func.call @mfma_index_B_16x16xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store ds_write_b64 data %tile addr %lds_addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok : !future_lds_write
  }
}
