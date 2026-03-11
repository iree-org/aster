// Kittens MFMA-aware LDS read primitives for 16x64_b tiles with v_mfma_f32_16x16x16 indexing.
//
// Maps MFMA A/B fragment indices to XOR-swizzled byte addresses in LDS.
// elt_size parameter allows reuse across f16 (2 bytes), fp8 (1 byte), etc.

// Register types
!vx2 = !amdgcn.vgpr<[? + 2]>

// Descriptor type
!index_pair = !aster_utils.struct<i: index, j: index>

// Future type
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.library @kittens_lds_mfma_16x64_b isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @mfma_index_A_16x16_f16() -> !index_pair
  func.func private @mfma_index_B_16x16_f16() -> !index_pair
  // From lds_16x64_b.mlir
  func.func private @read_vx2_from_lds(index, index, index) -> !future_lds_read
  func.func private @read_vx2_from_lds_at(index) -> !future_lds_read

  // From indexing.mlir
  func.func private @lds_swizzled_addr_16x64_b(index, index, index) -> index

  // Compute the swizzled LDS read address for MFMA A fragment.
  // Returns index that can be placed anywhere in the schedule.
  func.func private @compute_lds_read_addr_A(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> index {
    %mfma_idx = func.call @mfma_index_A_16x16_f16() : () -> !index_pair
    %row, %col_elt = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %byte_in_row = affine.apply affine_map<()[ce, koff, eb] -> (ce * eb + koff)>()
        [%col_elt, %k_byte_offset, %elt_size]
    %addr = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    return %addr : index
  }

  // Compute the swizzled LDS read address for MFMA B fragment.
  // Returns index that can be placed anywhere in the schedule.
  func.func private @compute_lds_read_addr_B(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> index {
    %mfma_idx = func.call @mfma_index_B_16x16_f16() : () -> !index_pair
    %col_elt, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %byte_in_row = affine.apply affine_map<()[ce, koff, eb] -> (ce * eb + koff)>()
        [%col_elt, %k_byte_offset, %elt_size]
    %addr = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    return %addr : index
  }

  // MFMA-aware LDS read: maps MFMA A fragment index to swizzled byte address.
  // elt_size: bytes per element (e.g. 2 for f16, 1 for fp8).
  func.func private @load_lds_A_swizzled(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> !future_lds_read {
    %addr = func.call @compute_lds_read_addr_A(%tile_base, %k_byte_offset, %elt_size)
        : (index, index, index) -> index
    %fut = func.call @read_vx2_from_lds_at(%addr)
        : (index) -> !future_lds_read
    return %fut : !future_lds_read
  }

  // MFMA-aware LDS read: maps MFMA B fragment index to swizzled byte address.
  // elt_size: bytes per element (e.g. 2 for f16, 1 for fp8).
  func.func private @load_lds_B_swizzled(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> !future_lds_read {
    %addr = func.call @compute_lds_read_addr_B(%tile_base, %k_byte_offset, %elt_size)
        : (index, index, index) -> index
    %fut = func.call @read_vx2_from_lds_at(%addr)
        : (index) -> !future_lds_read
    return %fut : !future_lds_read
  }

}
