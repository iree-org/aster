// Kittens LDS primitives for 32x32 f16 tiles (feeding 32x32x8 MFMA).
//
// All transfers use 32x32 tile granularity for coalesced global memory access.
// Thread mapping for cooperative fill (64 lanes, 32x32 tile):
//   row_in_group = lane_id / 4 (0..15), col = (lane_id % 4) * 8 (0,8,16,24)
//   Each thread loads 8 consecutive f16 values (16 bytes = dwordx4).
//   4 threads per row -> 32 f16 per row -> full 64-byte cache line coalescing.
//   2 load cycles per thread to cover 32 rows (groups of 16).
//
// LDS layout: 32x32 row-major with XOR swizzle, stride = 64 bytes per row.
// Total: 2048 bytes per tile. 4 MFMAs per tile (32x32x8 each).
//
// XOR swizzle for bank-conflict-free LDS access:
//   swizzled_byte = byte_in_row XOR (((row / 2) % 8) * 8)
//   Permutes bits [5:3] of byte_in_row based on row index.
//   Achieves optimal 2-way bank conflicts for MFMA ds_read_b64 (theoretical min).
//   Zero bank conflicts on cooperative fill writes.
//
// Global loads use dwordx4 (16 bytes/thread, 2 loads per tile).
// LDS writes split each dwordx4 into 2 x ds_write_b64 with independent swizzle.
//
// MFMA A/B fragment read from 32x32 swizzled LDS layout:
//   For sub-tile k (0..3), K cols k*8..k*8+7:
//     byte_in_row = k*16 + mfma_col * 2
//     swizzled_byte = byte_in_row XOR (((mfma_row / 2) % 8) * 8)
//     offset = base + mfma_row * 64 + swizzled_byte

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!vx16 = !amdgcn.vgpr<[? + 16]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx16

// Future/token types
!write_token = !amdgcn.write_token<flat>
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

amdgcn.library @kittens_lds_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @mfma_index_A_32x32xf16() -> !index_pair
  func.func private @mfma_index_B_32x32xf16() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4

  // From futures.mlir
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  // From global_32x32_f16.mlir
  func.func private @mfma_f32_32x32x8_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

  //===--------------------------------------------------------------------===//
  // Future value extraction for dwordx4 global loads
  //===--------------------------------------------------------------------===//

  // Wait on a global read future and extract the vx4 value.
  func.func private @get_global_load_value_vx4(%future: !future_global_read) -> !vx4 {
    %value_any, %token = aster_utils.struct_extract %future ["value", "token"]
        : !future_global_read -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.wait deps %token : !amdgcn.read_token<flat>
    %value = aster_utils.from_any %value_any : !vx4
    return %value : !vx4
  }

  //===--------------------------------------------------------------------===//
  // XOR swizzle helper
  //===--------------------------------------------------------------------===//

  // Compute swizzled LDS address: base + row*64 + (byte_in_row XOR mask).
  // mask = ((row / 2) % 8) * 8 -- permutes bits [5:3] based on row.
  // Uses arith.xori since affine_map cannot express XOR.
  func.func private @lds_swizzle_addr(%base: index, %row: index, %byte_in_row: index) -> index {
    %mask = affine.apply affine_map<(r) -> (((r floordiv 2) mod 8) * 8)>(%row)
    %mask_i32 = arith.index_cast %mask : index to i32
    %byte_i32 = arith.index_cast %byte_in_row : index to i32
    %swizzled_i32 = arith.xori %byte_i32, %mask_i32 : i32
    %swizzled = arith.index_cast %swizzled_i32 : i32 to index
    %addr = affine.apply affine_map<()[b, r, s] -> (b + r * 64 + s)>()[%base, %row, %swizzled]
    return %addr : index
  }

  //===--------------------------------------------------------------------===//
  // Thread mapping for 32x32 cooperative fill (dwordx4)
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row_in_group, col) for 32x32 tile cooperative fill.
  // 64 lanes, 4 threads per row, 8 f16 elements per thread (dwordx4):
  //   row_in_group = lane_id / 4 (0..15)
  //   col = (lane_id % 4) * 8 (0, 8, 16, 24)
  // 2 load/store cycles cover all 32 rows (row groups 0-15, 16-31).
  func.func private @thread_tile_pos_32x32() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 4) * 8)>()[%lane]
    return %row, %col : index, index
  }

  //===--------------------------------------------------------------------===//
  // Global Load (32x32 tile, 2 coalesced dwordx4 row-group loads)
  //===--------------------------------------------------------------------===//

  // Issue 2 global loads for a 32x32 f16 tile with coalesced access.
  // Each load covers 16 rows (4 threads/row * 16 bytes/thread = full 64-byte row).
  // Returns 2 futures: row groups 0-15 and 16-31.
  // Global memory is NOT swizzled -- coalesced row-major layout.
  func.func private @load_global_tile_32x32_f16(
      %ptr: !sx2, %m: index, %k_base: index, %stride: index
  ) -> (!future_global_read, !future_global_read) {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32

    // Row group 0: rows 0-15
    %desc0 = aster_utils.struct_create(%m, %k_base, %row_in_group, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off0 = func.call @tiled_matrix_offset(%desc0) : (!index_descriptor_2level_2d) -> !v
    %tmp0 = func.call @alloc_vgprx4() : () -> !vx4
    %loaded0, %tok0 = amdgcn.load global_load_dwordx4 dest %tmp0 addr %ptr
        offset d(%off0) + c(%c0_i32) : dps(!vx4) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val0 = aster_utils.to_any %loaded0 : !vx4
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    // Row group 1: rows 16-31
    %row1 = affine.apply affine_map<()[rig] -> (rig + 16)>()[%row_in_group]
    %desc1 = aster_utils.struct_create(%m, %k_base, %row1, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off1 = func.call @tiled_matrix_offset(%desc1) : (!index_descriptor_2level_2d) -> !v
    %tmp1 = func.call @alloc_vgprx4() : () -> !vx4
    %loaded1, %tok1 = amdgcn.load global_load_dwordx4 dest %tmp1 addr %ptr
        offset d(%off1) + c(%c0_i32) : dps(!vx4) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val1 = aster_utils.to_any %loaded1 : !vx4
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    return %f0, %f1 : !future_global_read, !future_global_read
  }

  //===--------------------------------------------------------------------===//
  // LDS Store (32x32 tile, XOR-swizzled, from dwordx4 global loads)
  //===--------------------------------------------------------------------===//

  // Store 2 dwordx4 global load futures to LDS as a 32x32 XOR-swizzled tile.
  // Each future contains 16 bytes (vx4). Split into 2 x vx2 for swizzled writes.
  // Future 0: row group 0-15, Future 1: row group 16-31.
  // Returns 4 write tokens (2 per future, lo/hi halves).
  func.func private @store_global_tile_to_lds_32x32_f16(
      %lds_base: index,
      %gf0: !future_global_read, %gf1: !future_global_read
  ) -> (!future_lds_write, !future_lds_write, !future_lds_write, !future_lds_write) {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %c0_i32 = arith.constant 0 : i32
    // byte_in_row: lo = col*2, hi = col*2+8 (each vx2 covers 8 bytes)
    %byte_lo = affine.apply affine_map<(c) -> (c * 2)>(%col)
    %byte_hi = affine.apply affine_map<(c) -> (c * 2 + 8)>(%col)

    // Future 0: rows 0-15 -- split vx4 into lo/hi vx2
    %data0 = func.call @get_global_load_value_vx4(%gf0) : (!future_global_read) -> !vx4
    %r0:4 = amdgcn.split_register_range %data0 : !vx4
    %lo0 = amdgcn.make_register_range %r0#0, %r0#1 : !v, !v
    %hi0 = amdgcn.make_register_range %r0#2, %r0#3 : !v, !v

    %addr0_lo_idx = func.call @lds_swizzle_addr(%lds_base, %row_in_group, %byte_lo)
        : (index, index, index) -> index
    %addr0_lo_i32 = arith.index_cast %addr0_lo_idx : index to i32
    %addr0_lo = lsir.to_reg %addr0_lo_i32 : i32 -> !v
    %tok0 = amdgcn.store ds_write_b64 data %lo0 addr %addr0_lo offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    %addr0_hi_idx = func.call @lds_swizzle_addr(%lds_base, %row_in_group, %byte_hi)
        : (index, index, index) -> index
    %addr0_hi_i32 = arith.index_cast %addr0_hi_idx : index to i32
    %addr0_hi = lsir.to_reg %addr0_hi_i32 : i32 -> !v
    %tok1 = amdgcn.store ds_write_b64 data %hi0 addr %addr0_hi offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    // Future 1: rows 16-31 -- split vx4 into lo/hi vx2
    %data1 = func.call @get_global_load_value_vx4(%gf1) : (!future_global_read) -> !vx4
    %r1:4 = amdgcn.split_register_range %data1 : !vx4
    %lo1 = amdgcn.make_register_range %r1#0, %r1#1 : !v, !v
    %hi1 = amdgcn.make_register_range %r1#2, %r1#3 : !v, !v

    %row1 = affine.apply affine_map<()[rig] -> (rig + 16)>()[%row_in_group]
    %addr1_lo_idx = func.call @lds_swizzle_addr(%lds_base, %row1, %byte_lo)
        : (index, index, index) -> index
    %addr1_lo_i32 = arith.index_cast %addr1_lo_idx : index to i32
    %addr1_lo = lsir.to_reg %addr1_lo_i32 : i32 -> !v
    %tok2 = amdgcn.store ds_write_b64 data %lo1 addr %addr1_lo offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    %addr1_hi_idx = func.call @lds_swizzle_addr(%lds_base, %row1, %byte_hi)
        : (index, index, index) -> index
    %addr1_hi_i32 = arith.index_cast %addr1_hi_idx : index to i32
    %addr1_hi = lsir.to_reg %addr1_hi_i32 : i32 -> !v
    %tok3 = amdgcn.store ds_write_b64 data %hi1 addr %addr1_hi offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %tok0, %tok1, %tok2, %tok3
        : !future_lds_write, !future_lds_write, !future_lds_write, !future_lds_write
  }

  // Wait for 4 LDS write tokens.
  func.func private @wait_lds_writes_32x32(
      %t0: !future_lds_write, %t1: !future_lds_write,
      %t2: !future_lds_write, %t3: !future_lds_write
  ) {
    amdgcn.wait deps %t0 : !future_lds_write
    amdgcn.wait deps %t1 : !future_lds_write
    amdgcn.wait deps %t2 : !future_lds_write
    amdgcn.wait deps %t3 : !future_lds_write
    return
  }

  //===--------------------------------------------------------------------===//
  // LDS Read for MFMA (32x8 fragments from 32x32 XOR-swizzled LDS)
  //===--------------------------------------------------------------------===//

  // Read 4 MFMA A fragments from a 32x32 XOR-swizzled tile in LDS.
  // Sub-tile k (0..3) reads K cols k*8..k*8+7.
  // Thread t: mfma_row = t%32, mfma_col = (t/32)*4 (from mfma_index_A).
  // byte_in_row for sub-tile k = k*16 + mfma_col*2.
  // Swizzle applied via @lds_swizzle_addr.
  func.func private @load_lds_A_32x32_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : i32

    // Sub-tile 0: K cols 0-7, byte_in_row = col*2
    %byte0 = affine.apply affine_map<(c) -> (c * 2)>(%col)
    %off0_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte0)
        : (index, index, index) -> index
    %off0_i32 = arith.index_cast %off0_idx : index to i32
    %addr0 = lsir.to_reg %off0_i32 : i32 -> !v
    %dst0 = func.call @alloc_vgprx2() : () -> !vx2
    %result0, %tok0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val0 = aster_utils.to_any %result0 : !vx2
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 1: K cols 8-15, byte_in_row = 16 + col*2
    %byte1 = affine.apply affine_map<(c) -> (16 + c * 2)>(%col)
    %off1_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte1)
        : (index, index, index) -> index
    %off1_i32 = arith.index_cast %off1_idx : index to i32
    %addr1 = lsir.to_reg %off1_i32 : i32 -> !v
    %dst1 = func.call @alloc_vgprx2() : () -> !vx2
    %result1, %tok1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val1 = aster_utils.to_any %result1 : !vx2
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 2: K cols 16-23, byte_in_row = 32 + col*2
    %byte2 = affine.apply affine_map<(c) -> (32 + c * 2)>(%col)
    %off2_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte2)
        : (index, index, index) -> index
    %off2_i32 = arith.index_cast %off2_idx : index to i32
    %addr2 = lsir.to_reg %off2_i32 : i32 -> !v
    %dst2 = func.call @alloc_vgprx2() : () -> !vx2
    %result2, %tok2 = amdgcn.load ds_read_b64 dest %dst2 addr %addr2 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val2 = aster_utils.to_any %result2 : !vx2
    %f2 = aster_utils.struct_create(%val2, %tok2)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 3: K cols 24-31, byte_in_row = 48 + col*2
    %byte3 = affine.apply affine_map<(c) -> (48 + c * 2)>(%col)
    %off3_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte3)
        : (index, index, index) -> index
    %off3_i32 = arith.index_cast %off3_idx : index to i32
    %addr3 = lsir.to_reg %off3_i32 : i32 -> !v
    %dst3 = func.call @alloc_vgprx2() : () -> !vx2
    %result3, %tok3 = amdgcn.load ds_read_b64 dest %dst3 addr %addr3 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val3 = aster_utils.to_any %result3 : !vx2
    %f3 = aster_utils.struct_create(%val3, %tok3)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    return %f0, %f1, %f2, %f3
        : !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read
  }

  // Read 4 MFMA B fragments from a 32x32 XOR-swizzled tile in LDS.
  // Same swizzle formula as A, but using B indexing (reversed i/j extraction).
  func.func private @load_lds_B_32x32_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_B_32x32xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : i32

    // Sub-tile 0: K cols 0-7, byte_in_row = col*2
    %byte0 = affine.apply affine_map<(c) -> (c * 2)>(%col)
    %off0_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte0)
        : (index, index, index) -> index
    %off0_i32 = arith.index_cast %off0_idx : index to i32
    %addr0 = lsir.to_reg %off0_i32 : i32 -> !v
    %dst0 = func.call @alloc_vgprx2() : () -> !vx2
    %result0, %tok0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val0 = aster_utils.to_any %result0 : !vx2
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 1: K cols 8-15, byte_in_row = 16 + col*2
    %byte1 = affine.apply affine_map<(c) -> (16 + c * 2)>(%col)
    %off1_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte1)
        : (index, index, index) -> index
    %off1_i32 = arith.index_cast %off1_idx : index to i32
    %addr1 = lsir.to_reg %off1_i32 : i32 -> !v
    %dst1 = func.call @alloc_vgprx2() : () -> !vx2
    %result1, %tok1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val1 = aster_utils.to_any %result1 : !vx2
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 2: K cols 16-23, byte_in_row = 32 + col*2
    %byte2 = affine.apply affine_map<(c) -> (32 + c * 2)>(%col)
    %off2_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte2)
        : (index, index, index) -> index
    %off2_i32 = arith.index_cast %off2_idx : index to i32
    %addr2 = lsir.to_reg %off2_i32 : i32 -> !v
    %dst2 = func.call @alloc_vgprx2() : () -> !vx2
    %result2, %tok2 = amdgcn.load ds_read_b64 dest %dst2 addr %addr2 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val2 = aster_utils.to_any %result2 : !vx2
    %f2 = aster_utils.struct_create(%val2, %tok2)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 3: K cols 24-31, byte_in_row = 48 + col*2
    %byte3 = affine.apply affine_map<(c) -> (48 + c * 2)>(%col)
    %off3_idx = func.call @lds_swizzle_addr(%lds_base, %row, %byte3)
        : (index, index, index) -> index
    %off3_i32 = arith.index_cast %off3_idx : index to i32
    %addr3 = lsir.to_reg %off3_i32 : i32 -> !v
    %dst3 = func.call @alloc_vgprx2() : () -> !vx2
    %result3, %tok3 = amdgcn.load ds_read_b64 dest %dst3 addr %addr3 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val3 = aster_utils.to_any %result3 : !vx2
    %f3 = aster_utils.struct_create(%val3, %tok3)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    return %f0, %f1, %f2, %f3
        : !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read
  }

  //===--------------------------------------------------------------------===//
  // Compute (4 MFMAs from LDS read futures)
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // 16x16x16 MFMA variant: read 16x16 sub-tiles from 32x32 swizzled LDS
  //===--------------------------------------------------------------------===//
  //
  // A 32x32 LDS tile decomposes into 2x2 sub-tiles of 16x16:
  //   [m_sub][k_sub] for A, [n_sub][k_sub] for B (m_sub,n_sub,k_sub in {0,1}).
  // Each 16x16 sub-tile feeds one v_mfma_f32_16x16x16_f16.
  // Total: 8 MFMAs per pair of 32x32 tiles (2x2 output x 2 K passes).
  //
  // Output: 4 x vx4 sub-tile accumulators packed into vx16:
  //   [0:3]=C[0][0], [4:7]=C[0][1], [8:11]=C[1][0], [12:15]=C[1][1]

  // 16x16 MFMA indexing from common library
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair

  // Read 4 MFMA A fragments (16x16) from a 32x32 XOR-swizzled tile in LDS.
  // 2x2 loop over [ms, ks]: row = ms*16 + mfma_row, byte = ks*32 + mfma_col*2.
  // Returns: A[m0k0, m0k1, m1k0, m1k1].
  func.func private @load_lds_A_32x32_m16_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %mfma_row, %mfma_col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca() : memref<4x!future_lds_read>
    scf.for %ms = %c0 to %c2 step %c1 {
      scf.for %ks = %c0 to %c2 step %c1 {
        %row = affine.apply affine_map<(s, r) -> (s * 16 + r)>(%ms, %mfma_row)
        %byte = affine.apply affine_map<(s, c) -> (s * 32 + c * 2)>(%ks, %mfma_col)
        %off = func.call @lds_swizzle_addr(%lds_base, %row, %byte)
            : (index, index, index) -> index
        %off_i32 = arith.index_cast %off : index to i32
        %addr = lsir.to_reg %off_i32 : i32 -> !v
        %dst = func.call @alloc_vgprx2() : () -> !vx2
        %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
            : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
        %val = aster_utils.to_any %result : !vx2
        %f = aster_utils.struct_create(%val, %tok)
            : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
        %idx = affine.linearize_index [%ms, %ks] by (%c2, %c2) : index
        memref.store %f, %buf[%idx] : memref<4x!future_lds_read>
      } {aster.constexpr}
    } {aster.constexpr}

    %f0 = memref.load %buf[%c0] : memref<4x!future_lds_read>
    %f1 = memref.load %buf[%c1] : memref<4x!future_lds_read>
    %f2 = memref.load %buf[%c2] : memref<4x!future_lds_read>
    %f3 = memref.load %buf[%c3] : memref<4x!future_lds_read>
    return %f0, %f1, %f2, %f3
        : !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read
  }

  // Read 4 MFMA B fragments (16x16) from a 32x32 XOR-swizzled tile in LDS.
  // Same addressing as A -- MFMA handles B transpose internally.
  // Returns: B[n0k0, n0k1, n1k0, n1k1].
  func.func private @load_lds_B_32x32_m16_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %mfma_row, %mfma_col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca() : memref<4x!future_lds_read>
    scf.for %ns = %c0 to %c2 step %c1 {
      scf.for %ks = %c0 to %c2 step %c1 {
        %row = affine.apply affine_map<(s, r) -> (s * 16 + r)>(%ns, %mfma_row)
        %byte = affine.apply affine_map<(s, c) -> (s * 32 + c * 2)>(%ks, %mfma_col)
        %off = func.call @lds_swizzle_addr(%lds_base, %row, %byte)
            : (index, index, index) -> index
        %off_i32 = arith.index_cast %off : index to i32
        %addr = lsir.to_reg %off_i32 : i32 -> !v
        %dst = func.call @alloc_vgprx2() : () -> !vx2
        %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
            : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
        %val = aster_utils.to_any %result : !vx2
        %f = aster_utils.struct_create(%val, %tok)
            : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
        %idx = affine.linearize_index [%ns, %ks] by (%c2, %c2) : index
        memref.store %f, %buf[%idx] : memref<4x!future_lds_read>
      } {aster.constexpr}
    } {aster.constexpr}

    %f0 = memref.load %buf[%c0] : memref<4x!future_lds_read>
    %f1 = memref.load %buf[%c1] : memref<4x!future_lds_read>
    %f2 = memref.load %buf[%c2] : memref<4x!future_lds_read>
    %f3 = memref.load %buf[%c3] : memref<4x!future_lds_read>
    return %f0, %f1, %f2, %f3
        : !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read
  }

  // Chain 8 MFMAs (16x16x16): C[ms][ns] += A[ms][ks] * B[ns][ks] for ms,ns,ks in {0,1}.
  // A/B futures indexed as [ms*2+ks] and [ns*2+ks].
  // acc vx16 packed as 4 x vx4: C[0][0], C[0][1], C[1][0], C[1][1].
  func.func private @compute_mfmas_32x32_m16(
      %a0_fut: !future_lds_read, %a1_fut: !future_lds_read,
      %a2_fut: !future_lds_read, %a3_fut: !future_lds_read,
      %b0_fut: !future_lds_read, %b1_fut: !future_lds_read,
      %b2_fut: !future_lds_read, %b3_fut: !future_lds_read,
      %acc: !rt_C_f32
  ) -> !rt_C_f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Store A/B futures into indexable buffers
    %a_buf = memref.alloca() : memref<4x!future_lds_read>
    memref.store %a0_fut, %a_buf[%c0] : memref<4x!future_lds_read>
    memref.store %a1_fut, %a_buf[%c1] : memref<4x!future_lds_read>
    memref.store %a2_fut, %a_buf[%c2] : memref<4x!future_lds_read>
    memref.store %a3_fut, %a_buf[%c3] : memref<4x!future_lds_read>
    %b_buf = memref.alloca() : memref<4x!future_lds_read>
    memref.store %b0_fut, %b_buf[%c0] : memref<4x!future_lds_read>
    memref.store %b1_fut, %b_buf[%c1] : memref<4x!future_lds_read>
    memref.store %b2_fut, %b_buf[%c2] : memref<4x!future_lds_read>
    memref.store %b3_fut, %b_buf[%c3] : memref<4x!future_lds_read>

    // Split vx16 acc into 4 x vx4 sub-tile accumulators stored in buffer
    %r:16 = amdgcn.split_register_range %acc : !vx16
    %c_buf = memref.alloca() : memref<4x!vx4>
    %ci0 = amdgcn.make_register_range %r#0, %r#1, %r#2, %r#3 : !v, !v, !v, !v
    %ci1 = amdgcn.make_register_range %r#4, %r#5, %r#6, %r#7 : !v, !v, !v, !v
    %ci2 = amdgcn.make_register_range %r#8, %r#9, %r#10, %r#11 : !v, !v, !v, !v
    %ci3 = amdgcn.make_register_range %r#12, %r#13, %r#14, %r#15 : !v, !v, !v, !v
    memref.store %ci0, %c_buf[%c0] : memref<4x!vx4>
    memref.store %ci1, %c_buf[%c1] : memref<4x!vx4>
    memref.store %ci2, %c_buf[%c2] : memref<4x!vx4>
    memref.store %ci3, %c_buf[%c3] : memref<4x!vx4>

    // 8 MFMAs via 2x2x2 constexpr loop
    scf.for %ms = %c0 to %c2 step %c1 {
      scf.for %ns = %c0 to %c2 step %c1 {
        %c_idx = affine.linearize_index [%ms, %ns] by (%c2, %c2) : index
        scf.for %ks = %c0 to %c2 step %c1 {
          %a_idx = affine.linearize_index [%ms, %ks] by (%c2, %c2) : index
          %b_idx = affine.linearize_index [%ns, %ks] by (%c2, %c2) : index
          %af = memref.load %a_buf[%a_idx] : memref<4x!future_lds_read>
          %bf = memref.load %b_buf[%b_idx] : memref<4x!future_lds_read>
          %a_v = func.call @get_lds_read_value_vx2(%af) : (!future_lds_read) -> !vx2
          %b_v = func.call @get_lds_read_value_vx2(%bf) : (!future_lds_read) -> !vx2
          %c_old = memref.load %c_buf[%c_idx] : memref<4x!vx4>
          %c_new = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %c_old, %a_v, %b_v, %c_old
              : !vx2, !vx2, !vx4 -> !vx4
          memref.store %c_new, %c_buf[%c_idx] : memref<4x!vx4>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    // Recombine 4 x vx4 into vx16
    %r0 = memref.load %c_buf[%c0] : memref<4x!vx4>
    %r1 = memref.load %c_buf[%c1] : memref<4x!vx4>
    %r2 = memref.load %c_buf[%c2] : memref<4x!vx4>
    %r3 = memref.load %c_buf[%c3] : memref<4x!vx4>
    %s0:4 = amdgcn.split_register_range %r0 : !vx4
    %s1:4 = amdgcn.split_register_range %r1 : !vx4
    %s2:4 = amdgcn.split_register_range %r2 : !vx4
    %s3:4 = amdgcn.split_register_range %r3 : !vx4
    %result = amdgcn.make_register_range %s0#0, %s0#1, %s0#2, %s0#3, %s1#0, %s1#1, %s1#2, %s1#3, %s2#0, %s2#1, %s2#2, %s2#3, %s3#0, %s3#1, %s3#2, %s3#3 : !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v, !v

    return %result : !rt_C_f32
  }

  // Store a 32x32 f32 C tile from 16x16 MFMA layout to global memory.
  // vx16 packed as 4 x vx4: C[ms][ns] for ms,ns in {0,1}.
  // 16x16 C layout: col = lane%16, row_base = (lane/16)*4, 4 VGPRs per sub-tile.
  // 2x2 constexpr loop over sub-tiles, 4 rows each.
  func.func private @store_C_32x32_m16_f32(
      %tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index
  ) -> !write_token {
    %mfma_idx = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
    %col_sub, %row_base_sub = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %elt_size = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32

    // Split vx16 into 4 x vx4 sub-tiles stored in buffer
    %r:16 = amdgcn.split_register_range %tile : !vx16
    %sub_buf = memref.alloca() : memref<4x!vx4>
    %st0 = amdgcn.make_register_range %r#0, %r#1, %r#2, %r#3 : !v, !v, !v, !v
    %st1 = amdgcn.make_register_range %r#4, %r#5, %r#6, %r#7 : !v, !v, !v, !v
    %st2 = amdgcn.make_register_range %r#8, %r#9, %r#10, %r#11 : !v, !v, !v, !v
    %st3 = amdgcn.make_register_range %r#12, %r#13, %r#14, %r#15 : !v, !v, !v, !v
    memref.store %st0, %sub_buf[%c0] : memref<4x!vx4>
    memref.store %st1, %sub_buf[%c1] : memref<4x!vx4>
    memref.store %st2, %sub_buf[%c2] : memref<4x!vx4>
    memref.store %st3, %sub_buf[%c3] : memref<4x!vx4>

    // 2x2 loop over sub-tiles, 4 stores each
    %tok_buf = memref.alloca() : memref<1x!write_token>
    scf.for %ms = %c0 to %c2 step %c1 {
      scf.for %ns = %c0 to %c2 step %c1 {
        %sub_idx = affine.linearize_index [%ms, %ns] by (%c2, %c2) : index
        %sub_tile = memref.load %sub_buf[%sub_idx] : memref<4x!vx4>
        %sr:4 = amdgcn.split_register_range %sub_tile : !vx4
        %col = affine.apply affine_map<(s, c) -> (s * 16 + c)>(%ns, %col_sub)

        %row0 = affine.apply affine_map<(s, rb) -> (s * 16 + rb)>(%ms, %row_base_sub)
        %desc0 = aster_utils.struct_create(%m, %n, %row0, %col, %stride, %elt_size)
            : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
        %off0 = func.call @tiled_matrix_offset(%desc0) : (!index_descriptor_2level_2d) -> !v
        %tok0 = amdgcn.store global_store_dword data %sr#0 addr %ptr offset d(%off0) + c(%c0_i32)
            : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

        %row1 = affine.apply affine_map<(s, rb) -> (s * 16 + rb + 1)>(%ms, %row_base_sub)
        %desc1 = aster_utils.struct_create(%m, %n, %row1, %col, %stride, %elt_size)
            : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
        %off1 = func.call @tiled_matrix_offset(%desc1) : (!index_descriptor_2level_2d) -> !v
        %tok1 = amdgcn.store global_store_dword data %sr#1 addr %ptr offset d(%off1) + c(%c0_i32)
            : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

        %row2 = affine.apply affine_map<(s, rb) -> (s * 16 + rb + 2)>(%ms, %row_base_sub)
        %desc2 = aster_utils.struct_create(%m, %n, %row2, %col, %stride, %elt_size)
            : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
        %off2 = func.call @tiled_matrix_offset(%desc2) : (!index_descriptor_2level_2d) -> !v
        %tok2 = amdgcn.store global_store_dword data %sr#2 addr %ptr offset d(%off2) + c(%c0_i32)
            : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

        %row3 = affine.apply affine_map<(s, rb) -> (s * 16 + rb + 3)>(%ms, %row_base_sub)
        %desc3 = aster_utils.struct_create(%m, %n, %row3, %col, %stride, %elt_size)
            : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
        %off3 = func.call @tiled_matrix_offset(%desc3) : (!index_descriptor_2level_2d) -> !v
        %tok3 = amdgcn.store global_store_dword data %sr#3 addr %ptr offset d(%off3) + c(%c0_i32)
            : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

        memref.store %tok3, %tok_buf[%c0] : memref<1x!write_token>
      } {aster.constexpr}
    } {aster.constexpr}

    %last_tok = memref.load %tok_buf[%c0] : memref<1x!write_token>
    return %last_tok : !write_token
  }

  //===--------------------------------------------------------------------===//
  // Compute (4 MFMAs from LDS read futures, 32x32x8 variant)
  //===--------------------------------------------------------------------===//

  // Chain 4 MFMAs: extract values from LDS read futures and accumulate.
  func.func private @compute_mfmas_32x32(
      %a0_fut: !future_lds_read, %a1_fut: !future_lds_read,
      %a2_fut: !future_lds_read, %a3_fut: !future_lds_read,
      %b0_fut: !future_lds_read, %b1_fut: !future_lds_read,
      %b2_fut: !future_lds_read, %b3_fut: !future_lds_read,
      %acc: !rt_C_f32
  ) -> !rt_C_f32 {
    // MFMA 0
    %A0 = func.call @get_lds_read_value_vx2(%a0_fut) : (!future_lds_read) -> !vx2
    %B0 = func.call @get_lds_read_value_vx2(%b0_fut) : (!future_lds_read) -> !vx2
    %acc0 = func.call @mfma_f32_32x32x8_f16(%A0, %B0, %acc)
        : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

    // MFMA 1
    %A1 = func.call @get_lds_read_value_vx2(%a1_fut) : (!future_lds_read) -> !vx2
    %B1 = func.call @get_lds_read_value_vx2(%b1_fut) : (!future_lds_read) -> !vx2
    %acc1 = func.call @mfma_f32_32x32x8_f16(%A1, %B1, %acc0)
        : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

    // MFMA 2
    %A2 = func.call @get_lds_read_value_vx2(%a2_fut) : (!future_lds_read) -> !vx2
    %B2 = func.call @get_lds_read_value_vx2(%b2_fut) : (!future_lds_read) -> !vx2
    %acc2 = func.call @mfma_f32_32x32x8_f16(%A2, %B2, %acc1)
        : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

    // MFMA 3
    %A3 = func.call @get_lds_read_value_vx2(%a3_fut) : (!future_lds_read) -> !vx2
    %B3 = func.call @get_lds_read_value_vx2(%b3_fut) : (!future_lds_read) -> !vx2
    %acc3 = func.call @mfma_f32_32x32x8_f16(%A3, %B3, %acc2)
        : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

    return %acc3 : !rt_C_f32
  }
}
