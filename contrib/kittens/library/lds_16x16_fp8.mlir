// Kittens LDS primitives for 16x32 FP8 tiles (v_mfma_f32_16x16x32_fp8_fp8).
//
// Key insight: FP8 16x32 tiles have the same 32 data bytes per row as F16 16x16
// tiles (32 elements * 1 byte = 16 elements * 2 bytes = 32 bytes). So LDS
// allocation sizes, bank conflict avoidance, and physical transfer patterns
// are identical. Only addressing (element size) and MFMA indexing differ.
//
// Padded addressing: stride = 34 bytes/row (32 data + 2 padding).
// Tile size: 16 * 34 = 544 bytes (same as F16 padded).

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>

// Kittens register tile types for FP8
!rt_A_fp8 = !vx2
!rt_B_fp8 = !vx2

// Future types
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

amdgcn.library @kittens_lds_16x16_fp8 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @mfma_index_A_16x16xfp8() -> !index_pair
  func.func private @mfma_index_B_16x16xfp8() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @alloc_vgprx2() -> !vx2

  // From futures.mlir
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // LDS Allocation (544 bytes/tile, 34-byte padded stride)
  //===--------------------------------------------------------------------===//

  // 1-buffer: 2 tiles x 544 = 1,088 bytes total
  func.func private @alloc_lds_fp8_1buffer_padded() -> (index, index) {
    %A_base = amdgcn.alloc_lds 544
    %B_base = amdgcn.alloc_lds 544

    %A_off = amdgcn.get_lds_offset %A_base : index
    %B_off = amdgcn.get_lds_offset %B_base : index

    return %A_off, %B_off : index, index
  }

  //===--------------------------------------------------------------------===//
  // LDS Addressing
  //===--------------------------------------------------------------------===//

  // Padded addressing for FP8: stride = 34 bytes/row, element size = 1 byte.
  // offset = tile_base + row * 34 + col * 1
  func.func private @lds_element_offset_fp8_padded(
      %tile_base: index,
      %row: index,
      %col: index
  ) -> index {
    %offset = affine.apply affine_map<()[base, row, col] -> (base + row * 34 + col)>
        ()[%tile_base, %row, %col]
    return %offset : index
  }

  //===--------------------------------------------------------------------===//
  // Thread-to-Element Mapping
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row, col) for cooperative LDS loads of a 16x32 FP8 tile.
  // 64 lanes, 8 elements/lane, row-major: row = lane / 4, col = (lane % 4) * 8
  func.func private @thread_lds_slice_fp8() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 4) * 8)>()[%lane]
    return %row, %col : index, index
  }

  //===--------------------------------------------------------------------===//
  // Global -> LDS Transfer
  //===--------------------------------------------------------------------===//

  // 16x32 FP8 global load -> LDS using padded writes.
  // Each lane loads 8 bytes (8 fp8 elements) via global_load_dwordx2,
  // waits for data, then writes to LDS via ds_write_b64.
  func.func private @load_global_to_lds_fp8(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !future_lds_write {
    %row, %col = func.call @thread_lds_slice_fp8() : () -> (index, index)

    // Compute global memory offset: (m+row)*stride + (n+col)*1
    %elt_size = arith.constant 1 : index  // fp8 = 1 byte
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

    // Compute LDS address and write
    %lds_offset_idx = func.call @lds_element_offset_fp8_padded(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0_i32_2 = arith.constant 0 : i32
    %tok_lds = amdgcn.store ds_write_b64 data %loaded addr %lds_addr offset c(%c0_i32_2)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok_lds : !future_lds_write
  }

  //===--------------------------------------------------------------------===//
  // LDS -> Register (MFMA fragment reads)
  //===--------------------------------------------------------------------===//

  // A fragment: read from LDS using FP8 A indexing.
  // mfma_index_A_16x16xfp8 returns (i=row, j=col) where col=(lid/16)*8.
  func.func private @load_lds_A_fp8(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_A_16x16xfp8() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_fp8_padded(%lds_tile_base, %row, %col)
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

  // B fragment: read from LDS using FP8 B indexing.
  // mfma_index_B_16x16xfp8 returns (i=col, j=row); swap for row-major LDS.
  func.func private @load_lds_B_fp8(%lds_tile_base: index) -> !future_lds_read {
    %mfma_idx = func.call @mfma_index_B_16x16xfp8() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset_fp8_padded(%lds_tile_base, %row, %col)
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

  // Wait and extract tile values.
  func.func private @get_lds_A_fp8(%future: !future_lds_read) -> !rt_A_fp8 {
    %result = func.call @get_lds_read_value_vx2(%future)
        : (!future_lds_read) -> !vx2
    return %result : !rt_A_fp8
  }

  func.func private @get_lds_B_fp8(%future: !future_lds_read) -> !rt_B_fp8 {
    %result = func.call @get_lds_read_value_vx2(%future)
        : (!future_lds_read) -> !vx2
    return %result : !rt_B_fp8
  }
}
