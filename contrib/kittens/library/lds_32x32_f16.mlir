// Kittens LDS primitives for 32x32 f16 tiles (feeding 32x32x8 MFMA).
//
// All transfers use 32x32 tile granularity for coalesced global memory access.
// Thread mapping for cooperative fill (64 lanes, 32x32 tile):
//   row_in_group = lane_id / 8 (0..7), col = (lane_id % 8) * 4 (0,4,...,28)
//   Each thread loads/stores 4 consecutive f16 values (8 bytes = dwordx2).
//   8 threads per row -> 32 f16 per row -> full 64-byte cache line coalescing.
//   4 load/store cycles per thread to cover 32 rows (groups of 8).
//
// LDS layout: 32x32 row-major, stride = 64 bytes per row.
// Total: 2048 bytes per tile. 4 MFMAs per tile (32x32x8 each).
//
// MFMA A/B fragment read from 32x32 LDS layout:
//   For sub-tile k (0..3), K cols k*8..k*8+7:
//     offset = base + mfma_row * 64 + k*16 + mfma_col * 2

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
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

  // From futures.mlir
  func.func private @get_global_load_value_vx2(!future_global_read) -> !vx2
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  // From global_32x32_f16.mlir
  func.func private @mfma_f32_32x32x8_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

  //===--------------------------------------------------------------------===//
  // Thread mapping for 32x32 cooperative fill
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row_in_group, col) for 32x32 tile cooperative fill.
  // 64 lanes, 8 threads per row, 4 f16 elements per thread:
  //   row_in_group = lane_id / 8 (0..7)
  //   col = (lane_id % 8) * 4 (0, 4, 8, ..., 28)
  // 4 load/store cycles cover all 32 rows (row groups 0-7, 8-15, 16-23, 24-31).
  func.func private @thread_tile_pos_32x32() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 8)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 8) * 4)>()[%lane]
    return %row, %col : index, index
  }

  //===--------------------------------------------------------------------===//
  // Global Load (32x32 tile, 4 coalesced row-group loads)
  //===--------------------------------------------------------------------===//

  // Issue 4 global loads for a 32x32 f16 tile with coalesced access.
  // Each load covers 8 rows (8 threads/row * 4 f16/thread = full 32-element row).
  // Returns 4 futures: row groups 0-7, 8-15, 16-23, 24-31.
  func.func private @load_global_tile_32x32_f16(
      %ptr: !sx2, %m: index, %k_base: index, %stride: index
  ) -> (!future_global_read, !future_global_read, !future_global_read, !future_global_read) {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32

    // Row group 0: rows 0-7
    %desc0 = aster_utils.struct_create(%m, %k_base, %row_in_group, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off0 = func.call @tiled_matrix_offset(%desc0) : (!index_descriptor_2level_2d) -> !v
    %tmp0 = func.call @alloc_vgprx2() : () -> !vx2
    %loaded0, %tok0 = amdgcn.load global_load_dwordx2 dest %tmp0 addr %ptr
        offset d(%off0) + c(%c0_i32) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val0 = aster_utils.to_any %loaded0 : !vx2
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    // Row group 1: rows 8-15
    %row1 = affine.apply affine_map<()[rig] -> (rig + 8)>()[%row_in_group]
    %desc1 = aster_utils.struct_create(%m, %k_base, %row1, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off1 = func.call @tiled_matrix_offset(%desc1) : (!index_descriptor_2level_2d) -> !v
    %tmp1 = func.call @alloc_vgprx2() : () -> !vx2
    %loaded1, %tok1 = amdgcn.load global_load_dwordx2 dest %tmp1 addr %ptr
        offset d(%off1) + c(%c0_i32) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val1 = aster_utils.to_any %loaded1 : !vx2
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    // Row group 2: rows 16-23
    %row2 = affine.apply affine_map<()[rig] -> (rig + 16)>()[%row_in_group]
    %desc2 = aster_utils.struct_create(%m, %k_base, %row2, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off2 = func.call @tiled_matrix_offset(%desc2) : (!index_descriptor_2level_2d) -> !v
    %tmp2 = func.call @alloc_vgprx2() : () -> !vx2
    %loaded2, %tok2 = amdgcn.load global_load_dwordx2 dest %tmp2 addr %ptr
        offset d(%off2) + c(%c0_i32) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val2 = aster_utils.to_any %loaded2 : !vx2
    %f2 = aster_utils.struct_create(%val2, %tok2)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    // Row group 3: rows 24-31
    %row3 = affine.apply affine_map<()[rig] -> (rig + 24)>()[%row_in_group]
    %desc3 = aster_utils.struct_create(%m, %k_base, %row3, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off3 = func.call @tiled_matrix_offset(%desc3) : (!index_descriptor_2level_2d) -> !v
    %tmp3 = func.call @alloc_vgprx2() : () -> !vx2
    %loaded3, %tok3 = amdgcn.load global_load_dwordx2 dest %tmp3 addr %ptr
        offset d(%off3) + c(%c0_i32) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    %val3 = aster_utils.to_any %loaded3 : !vx2
    %f3 = aster_utils.struct_create(%val3, %tok3)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read

    return %f0, %f1, %f2, %f3
        : !future_global_read, !future_global_read, !future_global_read, !future_global_read
  }

  //===--------------------------------------------------------------------===//
  // LDS Store (32x32 tile, row-major, stride = 64 bytes/row)
  //===--------------------------------------------------------------------===//

  // Store 4 global load futures to LDS as a 32x32 row-major tile.
  // Future i contains data for row group i (rows i*8..i*8+7).
  // LDS layout: row r, col c -> offset = base + r * 64 + c * 2.
  func.func private @store_global_tile_to_lds_32x32_f16(
      %lds_base: index,
      %gf0: !future_global_read, %gf1: !future_global_read,
      %gf2: !future_global_read, %gf3: !future_global_read
  ) -> (!future_lds_write, !future_lds_write, !future_lds_write, !future_lds_write) {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %c0_i32 = arith.constant 0 : i32

    // Row group 0: rows 0-7
    %loaded0 = func.call @get_global_load_value_vx2(%gf0) : (!future_global_read) -> !vx2
    %off0 = affine.apply affine_map<()[base, rig, col] -> (base + rig * 64 + col * 2)>
        ()[%lds_base, %row_in_group, %col]
    %off0_i32 = arith.index_cast %off0 : index to i32
    %addr0 = lsir.to_reg %off0_i32 : i32 -> !v
    %tok0 = amdgcn.store ds_write_b64 data %loaded0 addr %addr0 offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    // Row group 1: rows 8-15
    %loaded1 = func.call @get_global_load_value_vx2(%gf1) : (!future_global_read) -> !vx2
    %off1 = affine.apply affine_map<()[base, rig, col] -> (base + (rig + 8) * 64 + col * 2)>
        ()[%lds_base, %row_in_group, %col]
    %off1_i32 = arith.index_cast %off1 : index to i32
    %addr1 = lsir.to_reg %off1_i32 : i32 -> !v
    %tok1 = amdgcn.store ds_write_b64 data %loaded1 addr %addr1 offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    // Row group 2: rows 16-23
    %loaded2 = func.call @get_global_load_value_vx2(%gf2) : (!future_global_read) -> !vx2
    %off2 = affine.apply affine_map<()[base, rig, col] -> (base + (rig + 16) * 64 + col * 2)>
        ()[%lds_base, %row_in_group, %col]
    %off2_i32 = arith.index_cast %off2 : index to i32
    %addr2 = lsir.to_reg %off2_i32 : i32 -> !v
    %tok2 = amdgcn.store ds_write_b64 data %loaded2 addr %addr2 offset c(%c0_i32)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    // Row group 3: rows 24-31
    %loaded3 = func.call @get_global_load_value_vx2(%gf3) : (!future_global_read) -> !vx2
    %off3 = affine.apply affine_map<()[base, rig, col] -> (base + (rig + 24) * 64 + col * 2)>
        ()[%lds_base, %row_in_group, %col]
    %off3_i32 = arith.index_cast %off3 : index to i32
    %addr3 = lsir.to_reg %off3_i32 : i32 -> !v
    %tok3 = amdgcn.store ds_write_b64 data %loaded3 addr %addr3 offset c(%c0_i32)
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
  // LDS Read for MFMA (32x8 fragments from 32x32 row-major LDS)
  //===--------------------------------------------------------------------===//

  // Read 4 MFMA A fragments from a 32x32 tile stored row-major in LDS.
  // Sub-tile k (0..3) reads K cols k*8..k*8+7.
  // Thread t: mfma_row = t%32, mfma_col = (t/32)*4 (from mfma_index_A).
  // LDS offset for sub-tile k: base + mfma_row * 64 + k*16 + mfma_col * 2.
  func.func private @load_lds_A_32x32_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : i32

    // Sub-tile 0: K cols 0-7
    %off0 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + col * 2)>
        ()[%lds_base, %row, %col]
    %off0_i32 = arith.index_cast %off0 : index to i32
    %addr0 = lsir.to_reg %off0_i32 : i32 -> !v
    %dst0 = func.call @alloc_vgprx2() : () -> !vx2
    %result0, %tok0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val0 = aster_utils.to_any %result0 : !vx2
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 1: K cols 8-15 (+16 bytes within each row)
    %off1 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 16 + col * 2)>
        ()[%lds_base, %row, %col]
    %off1_i32 = arith.index_cast %off1 : index to i32
    %addr1 = lsir.to_reg %off1_i32 : i32 -> !v
    %dst1 = func.call @alloc_vgprx2() : () -> !vx2
    %result1, %tok1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val1 = aster_utils.to_any %result1 : !vx2
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 2: K cols 16-23 (+32 bytes)
    %off2 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 32 + col * 2)>
        ()[%lds_base, %row, %col]
    %off2_i32 = arith.index_cast %off2 : index to i32
    %addr2 = lsir.to_reg %off2_i32 : i32 -> !v
    %dst2 = func.call @alloc_vgprx2() : () -> !vx2
    %result2, %tok2 = amdgcn.load ds_read_b64 dest %dst2 addr %addr2 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val2 = aster_utils.to_any %result2 : !vx2
    %f2 = aster_utils.struct_create(%val2, %tok2)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 3: K cols 24-31 (+48 bytes)
    %off3 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 48 + col * 2)>
        ()[%lds_base, %row, %col]
    %off3_i32 = arith.index_cast %off3 : index to i32
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

  // Read 4 MFMA B fragments from a 32x32 tile stored row-major in LDS.
  // Same LDS offset formula as A, but using B indexing (reversed i/j extraction).
  func.func private @load_lds_B_32x32_f16(%lds_base: index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read) {
    %mfma_idx = func.call @mfma_index_B_32x32xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : i32

    // Sub-tile 0: K cols 0-7
    %off0 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + col * 2)>
        ()[%lds_base, %row, %col]
    %off0_i32 = arith.index_cast %off0 : index to i32
    %addr0 = lsir.to_reg %off0_i32 : i32 -> !v
    %dst0 = func.call @alloc_vgprx2() : () -> !vx2
    %result0, %tok0 = amdgcn.load ds_read_b64 dest %dst0 addr %addr0 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val0 = aster_utils.to_any %result0 : !vx2
    %f0 = aster_utils.struct_create(%val0, %tok0)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 1: K cols 8-15
    %off1 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 16 + col * 2)>
        ()[%lds_base, %row, %col]
    %off1_i32 = arith.index_cast %off1 : index to i32
    %addr1 = lsir.to_reg %off1_i32 : i32 -> !v
    %dst1 = func.call @alloc_vgprx2() : () -> !vx2
    %result1, %tok1 = amdgcn.load ds_read_b64 dest %dst1 addr %addr1 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val1 = aster_utils.to_any %result1 : !vx2
    %f1 = aster_utils.struct_create(%val1, %tok1)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 2: K cols 16-23
    %off2 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 32 + col * 2)>
        ()[%lds_base, %row, %col]
    %off2_i32 = arith.index_cast %off2 : index to i32
    %addr2 = lsir.to_reg %off2_i32 : i32 -> !v
    %dst2 = func.call @alloc_vgprx2() : () -> !vx2
    %result2, %tok2 = amdgcn.load ds_read_b64 dest %dst2 addr %addr2 offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %val2 = aster_utils.to_any %result2 : !vx2
    %f2 = aster_utils.struct_create(%val2, %tok2)
        : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read

    // Sub-tile 3: K cols 24-31
    %off3 = affine.apply affine_map<()[base, row, col] -> (base + row * 64 + 48 + col * 2)>
        ()[%lds_base, %row, %col]
    %off3_i32 = arith.index_cast %off3 : index to i32
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
