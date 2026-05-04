// Kittens LDS primitives for 16x64_b tiles (dwordx4 global loads, XOR swizzle).
//
// LDS layout: 16 rows x 64 bytes/row = 1024 bytes.
// K0 occupies bytes 0-31 per row, K1 occupies bytes 32-63 per row.
// XOR swizzle: mask = ((row/2)%8)*8, addr = tile_base + row*64 + (byte_in_row XOR mask).

// Register types
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

// Future types
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.library @kittens_lds_16x64_b isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @thread_tile_pos_16x64_b() -> (index, index)
  func.func private @lds_swizzled_addr_16x64_b(index, index, index) -> index
  // From futures.mlir
  func.func private @get_global_load_value_vx4(!future_global_read) -> !vx4
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From global_16x64_b[_buf].mlir
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read

  //===--------------------------------------------------------------------===//
  // Global -> LDS transfers for 16x64_b tiles
  //===--------------------------------------------------------------------===//

  // Write vx2 (8 bytes) to LDS at a pre-computed swizzled address (index).
  func.func private @write_vx2_to_lds_at(
      %data: !vx2, %addr_idx: index
  ) -> !future_lds_write {
    %addr_i32 = arith.index_cast %addr_idx : index to i32
    %lds_addr = lsir.to_reg %addr_i32 : i32 -> !v
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.ds_write_b64 data %data addr %lds_addr offset c(%c0) : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>
    return %tok : !future_lds_write
  }

  // Convenience wrapper: compute swizzled address + write in one call.
  func.func private @write_vx2_to_lds(
      %data: !vx2, %tile_base: index, %row: index, %byte_in_row: index
  ) -> !future_lds_write {
    %addr_idx = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    %tok = func.call @write_vx2_to_lds_at(%data, %addr_idx)
        : (!vx2, index) -> !future_lds_write
    return %tok : !future_lds_write
  }

  // Compute the two LDS write addresses for a 16x64_b tile write (lo and hi halves).
  // Returns two swizzled index addresses that can be placed anywhere in the schedule.
  func.func private @compute_lds_write_addrs_16x64_b(
      %tile_base: index
  ) -> (index, index) {
    %row, %col_byte = func.call @thread_tile_pos_16x64_b() : () -> (index, index)
    %col_byte_hi = affine.apply affine_map<()[c] -> (c + 8)>()[%col_byte]
    %addr_lo = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %col_byte)
        : (index, index, index) -> index
    %addr_hi = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %col_byte_hi)
        : (index, index, index) -> index
    return %addr_lo, %addr_hi : index, index
  }

  // Write vx4 (16 bytes) to LDS at pre-computed swizzled addresses (lo and hi).
  func.func private @write_vx4_to_lds_at(
      %loaded: !vx4, %addr_lo: index, %addr_hi: index
  ) -> (!future_lds_write, !future_lds_write) {
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %loaded : !vx4
    %lo = amdgcn.make_register_range %v0, %v1 : !v, !v
    %hi = amdgcn.make_register_range %v2, %v3 : !v, !v

    %tok_lo = func.call @write_vx2_to_lds_at(%lo, %addr_lo)
        : (!vx2, index) -> !future_lds_write
    %tok_hi = func.call @write_vx2_to_lds_at(%hi, %addr_hi)
        : (!vx2, index) -> !future_lds_write

    return %tok_lo, %tok_hi : !future_lds_write, !future_lds_write
  }

  // Convenience wrapper: compute addresses + write in one call.
  func.func private @write_vx4_to_lds_16x64_b(
      %tile_base: index, %loaded: !vx4
  ) -> (!future_lds_write, !future_lds_write) {
    %addr_lo, %addr_hi = func.call @compute_lds_write_addrs_16x64_b(%tile_base)
        : (index) -> (index, index)
    %tok_lo, %tok_hi = func.call @write_vx4_to_lds_at(%loaded, %addr_lo, %addr_hi)
        : (!vx4, index, index) -> (!future_lds_write, !future_lds_write)
    return %tok_lo, %tok_hi : !future_lds_write, !future_lds_write
  }

  // Wait for global load and write the full 16x64_b tile to LDS.
  // Each thread writes 16 bytes via 2 x ds_write_b64: lo at col_byte, hi at col_byte+8.
  func.func private @store_global_tile_to_lds_16x64_b(
      %tile_base: index, %global_future: !future_global_read
  ) -> (!future_lds_write, !future_lds_write) {
    %loaded = func.call @get_global_load_value_vx4(%global_future)
        : (!future_global_read) -> !vx4
    %tok_lo, %tok_hi = func.call @write_vx4_to_lds_16x64_b(%tile_base, %loaded)
        : (index, !vx4) -> (!future_lds_write, !future_lds_write)
    return %tok_lo, %tok_hi : !future_lds_write, !future_lds_write
  }

  //===--------------------------------------------------------------------===//
  // Read from LDS: 16x16 sub-tile (ds_read_b64) with K0/K1 selection
  //===--------------------------------------------------------------------===//
  // Read vx2 (8 bytes) from LDS at a pre-computed swizzled address (index).
  func.func private @read_vx2_from_lds_at(
      %addr_idx: index
  ) -> !future_lds_read {
    %addr_i32 = arith.index_cast %addr_idx : index to i32
    %lds_addr = lsir.to_reg %addr_i32 : i32 -> !v
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.ds_read_b64 dest %dst addr %lds_addr offset c(%c0) : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

  // Convenience wrapper: compute swizzled address + read in one call.
  func.func private @read_vx2_from_lds(
      %tile_base: index, %row: index, %byte_in_row: index
  ) -> !future_lds_read {
    %addr_idx = func.call @lds_swizzled_addr_16x64_b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    %future = func.call @read_vx2_from_lds_at(%addr_idx)
        : (index) -> !future_lds_read
    return %future : !future_lds_read
  }

}
