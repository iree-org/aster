// Ptr-based indexing functions for computing byte offsets and converting to VGPR.
//
// Offset computation stays as index for composability; @index_to_vgpr_i32
// converts the final result to a 32-bit !v for amdgcn.ptr_add or LDS addr.

!v = !amdgcn.vgpr
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>

amdgcn.library @common_indexing_ptr {
  // Compute linear byte offset for a 2-D matrix access.
  // Formula: i * stride + j * elt_size
  // Returns index (not !v) to preserve offset structure.
  func.func private @matrix_offset_idx(%desc: !index_descriptor_2d) -> index {
    %i, %j, %stride, %elt_size = aster_utils.struct_extract %desc
        ["i", "j", "stride", "elt_size_b"]
        : !index_descriptor_2d -> index, index, index, index
    %off = affine.apply
      affine_map<()[i, j, stride, elt_size] -> (i * stride + j * elt_size)>
      ()[%i, %j, %stride, %elt_size]
    return %off : index
  }

  // Compute total byte offset = tile-level offset + thread-level offset.
  // tile_desc: (tile_row, tile_col, stride, elt_size) for the tile-level position.
  // thread_off: pre-computed thread-level byte offset (from @matrix_offset_idx).
  func.func private @tile_thread_offset_idx(
      %tile_desc: !index_descriptor_2d, %thread_off: index) -> index {
    %tile_off = func.call @matrix_offset_idx(%tile_desc)
        : (!index_descriptor_2d) -> index
    %total = affine.apply affine_map<()[t, d] -> (t + d)>()[%tile_off, %thread_off]
    return %total : index
  }

  // Convert an index byte offset to a 32-bit VGPR for use as addr operand.
  // Truncates to i32: correct for AMDGCN byte offsets (max 4GB per pointer).
  func.func private @index_to_vgpr_i32(%off: index) -> !v {
    %i32 = arith.index_cast %off : index to i32
    %reg = lsir.to_reg %i32 : i32 -> !v
    return %reg : !v
  }
}
