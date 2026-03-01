// Kittens-style 16x16 tile abstractions for FP8 MFMA operations.
// Mirrors global_16x16_f16.mlir but for v_mfma_f32_16x16x32_fp8_fp8.
//
// Key differences from FP16:
//   - Element size: 1 byte (vs 2 for f16)
//   - MFMA K dimension: 32 (vs 16 for f16)
//   - Lane column group: 8 elements (vs 4 for f16)
//   - Same register sizes: A/B = !vx2 (8 bytes), C/D = !vx4 (4xf32)

// Register types from common library
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

// Token types for async memory operations
!write_token = !amdgcn.write_token<flat>

// Future types from futures.mlir
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

// Kittens register tile type aliases for 16x16x32 FP8 MFMA
//   !rt_A_fp8: A operand - 2 VGPRs holding 8xfp8 per thread (16x32 logical tile)
//   !rt_B_fp8: B operand - 2 VGPRs holding 8xfp8 per thread (16x32 logical tile)
//   !rt_C_f32: C/D operand - 4 VGPRs holding 4xf32 per thread (16x16 logical tile)
!rt_A_fp8 = !vx2
!rt_B_fp8 = !vx2
!rt_C_f32 = !vx4

amdgcn.library @kittens_tiles_16x16_fp8 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @mfma_index_A_16x16xfp8() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @matrix_offset(!index_descriptor_2d) -> !v
  // From futures.mlir
  func.func private @get_global_load_value_vx2(!future_global_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // Tile initialization functions
  //===--------------------------------------------------------------------===//

  // Initialize a 16x16 f32 accumulator tile to zero.
  func.func private @zero_C_fp8() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_vgprx4(%c0) : (i32) -> !vx4
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory load functions
  //===--------------------------------------------------------------------===//

  // Load a 16x32 fp8 tile from global memory in MFMA A fragment layout.
  // Each thread loads 8xfp8 (8 bytes) via global_load_dwordx2.
  // Lane i loads from row (i % 16), cols [(i/16)*8, (i/16)*8 + 8).
  //
  // Returns a future; call @get_A_fp8 to wait and extract the tile value.
  //
  // Parameters:
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  //
  // Returns: !future_global_read wrapping !rt_A_fp8
  func.func private @load_A_fp8(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !future_global_read {
    // Get MFMA A index for FP8: (row, col) where each lane handles 8 elements
    %mfma_idx = func.call @mfma_index_A_16x16xfp8() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte offset: (m + row) * stride + (n + col) * elt_size
    %elt_size = arith.constant 1 : index  // fp8 = 1 byte
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes (8xfp8) via global_load_dwordx2
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  // Wait for a pending A tile load and extract the value.
  func.func private @get_A_fp8(%future: !future_global_read) -> !rt_A_fp8 {
    %result = func.call @get_global_load_value_vx2(%future) : (!future_global_read) -> !vx2
    return %result : !rt_A_fp8
  }

  // Load a 16x32 fp8 tile from global memory in MFMA B fragment layout.
  // Uses the same physical layout as A since MFMA computes A @ B^T internally.
  //
  // Returns a future; call @get_B_fp8 to wait and extract the tile value.
  //
  // Parameters:
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  //
  // Returns: !future_global_read wrapping !rt_B_fp8
  func.func private @load_B_fp8(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !future_global_read {
    // B uses same physical layout as A for loading from row-major memory
    %mfma_idx = func.call @mfma_index_A_16x16xfp8() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte offset: (m + row) * stride + (n + col) * elt_size
    %elt_size = arith.constant 1 : index  // fp8 = 1 byte
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes (8xfp8) via global_load_dwordx2
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  // Wait for a pending B tile load and extract the value.
  func.func private @get_B_fp8(%future: !future_global_read) -> !rt_B_fp8 {
    %result = func.call @get_global_load_value_vx2(%future) : (!future_global_read) -> !vx2
    return %result : !rt_B_fp8
  }

  //===--------------------------------------------------------------------===//
  // MFMA operations
  //===--------------------------------------------------------------------===//

  // Perform a 16x16x32 FP8 matrix multiply-accumulate: D = A @ B^T + C
  // Uses v_mfma_f32_16x16x32_fp8_fp8 instruction.
  //
  // The MFMA computes: D[16x16] = A[16x32] @ B[16x32]^T + C[16x16]
  // where A and B are fp8 (E4M3), C and D are f32.
  //
  // Parameters:
  //   %A: !rt_A_fp8 - A operand (16x32 fp8 tile)
  //   %B: !rt_B_fp8 - B operand (16x32 fp8 tile, transposed internally)
  //   %C: !rt_C_f32 - C accumulator (16x16 f32 tile)
  //
  // Returns: !rt_C_f32 - D result (16x16 f32 tile)
  func.func private @mfma_f32_16x16x32_fp8_fp8(%A: !rt_A_fp8, %B: !rt_B_fp8, %C: !rt_C_f32) -> !rt_C_f32 {
    // Accumulator-in-place: reuse %C as DPS destination for loop compatibility.
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x32_fp8_fp8> %C, %A, %B, %C
        : !vx2, !vx2, !vx4 -> !vx4
    return %result : !rt_C_f32
  }

  // Future-accepting variant: waits for A and B loads, then performs MFMA.
  // Convenience wrapper for the common pattern of load -> wait -> mfma.
  func.func private @mfma_f32_16x16x32_fp8_fp8_future(%A_future: !future_global_read, %B_future: !future_global_read, %C: !rt_C_f32) -> !rt_C_f32 {
    %A = func.call @get_A_fp8(%A_future) : (!future_global_read) -> !rt_A_fp8
    %B = func.call @get_B_fp8(%B_future) : (!future_global_read) -> !rt_B_fp8
    %result = func.call @mfma_f32_16x16x32_fp8_fp8(%A, %B, %C) : (!rt_A_fp8, !rt_B_fp8, !rt_C_f32) -> !rt_C_f32
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory store functions for C tile
  //===--------------------------------------------------------------------===//

  // Store a 16x16 f32 C tile to global memory from MFMA C fragment layout.
  // Identical to FP16 version - C layout does not change between FP8 and FP16.
  //
  // Note: Returns the last write token. The 3 earlier store tokens are not returned,
  // but WaitAnalysis will conservatively insert proper s_waitcnt instructions
  // for all outstanding stores when this token is waited on.
  func.func private @store_C_fp8(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) -> !write_token {
    %mfma_idx = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %elt_size = arith.constant 4 : index  // f32 = 4 bytes
    %c0_i32 = arith.constant 0 : i32

    // Split the !vx4 into 4 individual f32 values
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %tile : !vx4

    // Store each of the 4 f32 values at consecutive row positions, same column

    // Value 0: at row_base + 0
    %c0 = arith.constant 0 : index
    %row0 = arith.addi %row_base, %c0 : index
    %desc0 = aster_utils.struct_create(%m, %n, %row0, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off0 = func.call @tiled_matrix_offset(%desc0) : (!index_descriptor_2level_2d) -> !v
    %tok0 = amdgcn.store global_store_dword data %v0 addr %ptr offset d(%off0) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    // Value 1: at row_base + 1
    %c1 = arith.constant 1 : index
    %row1 = arith.addi %row_base, %c1 : index
    %desc1 = aster_utils.struct_create(%m, %n, %row1, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off1 = func.call @tiled_matrix_offset(%desc1) : (!index_descriptor_2level_2d) -> !v
    %tok1 = amdgcn.store global_store_dword data %v1 addr %ptr offset d(%off1) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    // Value 2: at row_base + 2
    %c2 = arith.constant 2 : index
    %row2 = arith.addi %row_base, %c2 : index
    %desc2 = aster_utils.struct_create(%m, %n, %row2, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off2 = func.call @tiled_matrix_offset(%desc2) : (!index_descriptor_2level_2d) -> !v
    %tok2 = amdgcn.store global_store_dword data %v2 addr %ptr offset d(%off2) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    // Value 3: at row_base + 3
    %c3 = arith.constant 3 : index
    %row3 = arith.addi %row_base, %c3 : index
    %desc3 = aster_utils.struct_create(%m, %n, %row3, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off3 = func.call @tiled_matrix_offset(%desc3) : (!index_descriptor_2level_2d) -> !v
    %tok3 = amdgcn.store global_store_dword data %v3 addr %ptr offset d(%off3) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    return %tok3 : !write_token
  }
}
