// Kittens LDS primitives for 16x64_b tiles (dwordx4 global loads, XOR swizzle).
//
// LDS layout: 16 rows x 64 bytes/row = 1024 bytes.
// K0 occupies bytes 0-31 per row, K1 occupies bytes 32-63 per row.
// XOR swizzle: mask = ((row/2)%8)*8, addr = tile_base + row*64 + (byte_in_row XOR mask).

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
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
  func.func private @thread_tile_pos_16x64b() -> (index, index)
  func.func private @lds_swizzled_addr_16x64b(index, index, index) -> index
  // From futures.mlir
  func.func private @get_global_load_value_vx4(!future_global_read) -> !vx4
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From global_16x64_b.mlir
  func.func private @load_global_tile_16x64_b(!sx2, index, index, index) -> !future_global_read

  //===--------------------------------------------------------------------===//
  // Global -> LDS transfers for 16x64_b tiles
  //===--------------------------------------------------------------------===//

  // Write vx2 (8 bytes) to XOR-swizzled LDS at (row, byte_in_row).
  func.func private @write_vx2_to_lds(
      %data: !vx2, %tile_base: index, %row: index, %byte_in_row: index
  ) -> !future_lds_write {
    %addr_idx = func.call @lds_swizzled_addr_16x64b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    %addr_i32 = arith.index_cast %addr_idx : index to i32
    %lds_addr = lsir.to_reg %addr_i32 : i32 -> !v
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store ds_write_b64 data %data addr %lds_addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    return %tok : !future_lds_write
  }

  // Write vx4 (16 bytes) to LDS as two bank-conflict-free swizzled ds_write_b64 at
  //   - (row, byte_in_row) and
  //   - (row, byte_in_row + 8)
  // Core write primitive: caller is responsible for awaiting the global load and extracting the vx4.
  func.func private @write_vx4_to_lds_16x64_b(
      %tile_base: index, %loaded: !vx4
  ) -> (!future_lds_write, !future_lds_write) {
    // Split vx4: lo = bytes 0-7, hi = bytes 8-15.
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %loaded : !vx4
    %lo = amdgcn.make_register_range %v0, %v1 : !v, !v
    %hi = amdgcn.make_register_range %v2, %v3 : !v, !v

    %row, %col_byte = func.call @thread_tile_pos_16x64b() : () -> (index, index)
    %col_byte_hi = affine.apply affine_map<()[c] -> (c + 8)>()[%col_byte]

    %tok_lo = func.call @write_vx2_to_lds(%lo, %tile_base, %row, %col_byte)
        : (!vx2, index, index, index) -> !future_lds_write
    %tok_hi = func.call @write_vx2_to_lds(%hi, %tile_base, %row, %col_byte_hi)
        : (!vx2, index, index, index) -> !future_lds_write

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
  // Core: read vx2 (8 bytes) from XOR-swizzled LDS at (row, byte_in_row).
  func.func private @read_vx2_from_lds(
      %tile_base: index, %row: index, %byte_in_row: index
  ) -> !future_lds_read {
    %addr_idx = func.call @lds_swizzled_addr_16x64b(%tile_base, %row, %byte_in_row)
        : (index, index, index) -> index
    %addr_i32 = arith.index_cast %addr_idx : index to i32
    %lds_addr = lsir.to_reg %addr_i32 : i32 -> !v
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
    return %future : !future_lds_read
  }

}
