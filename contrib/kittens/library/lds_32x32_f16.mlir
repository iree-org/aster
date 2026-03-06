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
