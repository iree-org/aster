// Kittens MFMA-aware LDS read primitives for 32x32x8 f16 MFMA indexing.
//
// Maps v_mfma_f32_32x32x8_f16 A/B fragment indices to XOR-swizzled byte
// addresses in LDS, using @lds_swizzled_addr_32x32 (stride = 64 bytes/row).
//
// LDS layout: a 32-row tile is stored as two consecutive 16x64_b slots:
//   rows 0-15  at tile_base + 0    (upper half, 1024 bytes)
//   rows 16-31 at tile_base + 1024 (lower half, 1024 bytes)
// Since @lds_swizzled_addr_32x32 uses base + row*64 + swizzle(row, byte),
// and the swizzle mask ((row/2)%8)*8 repeats identically for rows 0-15 and 16-31,
// a single call with row in {0..31} and the unified tile_base correctly addresses
// both halves without any special-casing.
//
// For v_mfma_f32_32x32x8_f16 (K=8 per step, 4 f16 per lane):
//   A fragment: lane l holds row l%32, cols [(l/32)*4 : (l/32)*4+4] in f16.
//   B fragment: same physical layout (transposed in semantics).
//   k_byte_offset = k_step * 16  (8 f16 * 2 bytes = 16 bytes per MFMA step).
//   lane l reads at byte_in_row = k_byte_offset + (l/32)*8.

// Register types
!vx2 = !amdgcn.vgpr<[? + 2]>

// Descriptor type
!index_pair = !aster_utils.struct<i: index, j: index>

// Future type
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.library @kittens_lds_mfma_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @mfma_index_A_32x32_f16() -> !index_pair
  func.func private @mfma_index_B_32x32_f16() -> !index_pair
  func.func private @lds_swizzled_addr_32x32(index, index, index) -> index
  // From lds_16x64_b.mlir (same ds_read_b64 primitive, reused for 32x32)
  func.func private @read_vx2_from_lds_at(index) -> !future_lds_read

  // Compute the swizzled LDS read address for a 32x32x8 MFMA A fragment.
  // tile_base: byte offset to the start of the 32-row LDS tile.
  // k_byte_offset: byte offset within a row for this MFMA step (= k_step * 16).
  // elt_size: bytes per f16 element (2).
  func.func private @compute_lds_read_addr_A_32x32(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> index {
    // A fragment: row = lane%32, col_elt = (lane/32)*4.
    // byte_in_row = col_elt * elt_size + k_byte_offset.
    %mfma_idx = func.call @mfma_index_A_32x32_f16() : () -> !index_pair
    %row, %col_elt = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %byte_in_row = affine.apply affine_map<()[ce, koff, eb] -> (ce * eb + koff)>()
        [%col_elt, %k_byte_offset, %elt_size]
    %addr = func.call @lds_swizzled_addr_32x32(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    return %addr : index
  }

  // Compute the swizzled LDS read address for a 32x32x8 MFMA B fragment.
  // Same physical layout as A (B is transposed in MFMA semantics).
  func.func private @compute_lds_read_addr_B_32x32(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> index {
    // B fragment: col = lane%32, row_elt = (lane/32)*4.
    // byte_in_row = row_elt * elt_size + k_byte_offset.
    %mfma_idx = func.call @mfma_index_B_32x32_f16() : () -> !index_pair
    %col_elt, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %byte_in_row = affine.apply affine_map<()[ce, koff, eb] -> (ce * eb + koff)>()
        [%col_elt, %k_byte_offset, %elt_size]
    %addr = func.call @lds_swizzled_addr_32x32(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    return %addr : index
  }

  // MFMA-aware LDS read: issue ds_read_b64 for the A fragment of v_mfma_f32_32x32x8_f16.
  // tile_base: byte offset to the unified 32-row LDS tile (upper half at tile_base,
  //            lower half at tile_base+1024, addressed as rows 0..31).
  // k_byte_offset: byte offset within the row for this MFMA step (= k_step * 16).
  // elt_size: bytes per f16 element (2).
  func.func private @load_lds_A_swizzled_32x32(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> !future_lds_read {
    %addr = func.call @compute_lds_read_addr_A_32x32(%tile_base, %k_byte_offset, %elt_size)
        : (index, index, index) -> index
    %fut = func.call @read_vx2_from_lds_at(%addr)
        : (index) -> !future_lds_read
    return %fut : !future_lds_read
  }

  // MFMA-aware LDS read: issue ds_read_b64 for the B fragment of v_mfma_f32_32x32x8_f16.
  func.func private @load_lds_B_swizzled_32x32(
      %tile_base: index, %k_byte_offset: index, %elt_size: index
  ) -> !future_lds_read {
    %addr = func.call @compute_lds_read_addr_B_32x32(%tile_base, %k_byte_offset, %elt_size)
        : (index, index, index) -> index
    %fut = func.call @read_vx2_from_lds_at(%addr)
        : (index) -> !future_lds_read
    return %fut : !future_lds_read
  }
}
