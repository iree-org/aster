// Ptr-based indexing functions for computing byte offsets and converting to VGPR.
//
// Offset computation stays as index for composability; @index_to_vgpr_i32
// converts the final result to a 32-bit !v for amdgcn.ptr_add or LDS addr.

!s = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx4 = !amdgcn.sgpr<[? + 4]>
!v = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!gptr = !ptr.ptr<#amdgcn.addr_space<global, read_write>>
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

  // Compute a global address from an SGPR base pointer and an index byte offset.
  // Converts the SGPR pair to a generic pointer, applies the offset via ptr.ptr_add
  // so aster-optimize-ptr-add can decompose it into const/uniform/dynamic components,
  // and returns a !vx2 (VGPR pair) ready for amdgcn.load/store addr operand.
  func.func private @global_addr_from_offset(
      %base: !sx2, %byte_off: index) -> !vx2 {
    %gptr = lsir.from_reg %base : !sx2 -> !gptr
    %off_i32 = arith.index_cast %byte_off : index to i32
    %addr_ptr = ptr.ptr_add %gptr, %off_i32 : !gptr, i32
    %addr = lsir.to_reg %addr_ptr : !gptr -> !vx2
    return %addr : !vx2
  }

  // Construct a raw buffer resource descriptor (SRD) from an SGPR base pointer.
  // Uses stride=0 (raw/OFFEN mode) and num_records=0xFFFFFFFF (no bounds check).
  // flags=131072 (0x20000) sets DATA_FORMAT=4 (32-bit), required for GFX9 buffer ops.
  func.func private @make_raw_buffer_rsrc(%base: !sx2) -> !sx4 {
    %max_bytes = arith.constant -1 : i32
    %num_records = lsir.to_reg %max_bytes : i32 -> !s
    %c0_stride = arith.constant 0 : i32
    %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %c0_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!sx2, !s, i32) -> !sx4
    return %rsrc : !sx4
  }
}
