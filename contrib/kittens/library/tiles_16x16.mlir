// Kittens-style 16x16 tile abstractions for MFMA operations.
// Provides high-level primitives for register tiles used in GEMM kernels.

// Register types from common library
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

// Kittens register tile type aliases for 16x16x16 MFMA
//   !rt_A_f16: A operand - 2 VGPRs holding 4xf16 per thread (16x16 logical tile)
//   !rt_B_f16: B operand - 2 VGPRs holding 4xf16 per thread (16x16 logical tile)
//   !rt_C_f32: C/D operand - 4 VGPRs holding 4xf32 per thread (16x16 logical tile)
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4

amdgcn.library @kittens_tiles_16x16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @matrix_offset(!index_descriptor_2d) -> !v

  //===--------------------------------------------------------------------===//
  // Tile initialization functions
  //===--------------------------------------------------------------------===//

  // Initialize a 16x16 f32 accumulator tile to zero.
  // Returns a !rt_C_f32 (= !vx4) with all elements set to 0.
  func.func private @zero_C() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_vgprx4(%c0) : (i32) -> !vx4
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory load functions (Phase 2)
  //===--------------------------------------------------------------------===//

  // Load a 16x16 f16 tile from global memory in MFMA A fragment layout.
  // Each thread loads 4xf16 (8 bytes) via global_load_dwordx2.
  // Lane i loads from row (i % 16), cols [(i/16)*4, (i/16)*4 + 4).
  //
  // Parameters:
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  //
  // Returns: !rt_A_f16 (= !vx2) with loaded data
  func.func private @load_A_f16(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !rt_A_f16 {
    // Get MFMA A index for this lane: (row, col) where each lane handles 4 elements
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte offset: (m + row) * stride + (n + col) * elt_size
    %elt_size = arith.constant 2 : index  // f16 = 2 bytes
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes (4xf16) via global_load_dwordx2
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return %result : !rt_A_f16
  }

  // Load a 16x16 f16 tile from global memory in MFMA B fragment layout.
  // Uses the same physical layout as A since MFMA computes A @ B^T internally.
  // Each thread loads 4xf16 (8 bytes) via global_load_dwordx2.
  //
  // Parameters:
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  //
  // Returns: !rt_B_f16 (= !vx2) with loaded data
  func.func private @load_B_f16(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !rt_B_f16 {
    // B uses same physical layout as A for loading from row-major memory
    // (MFMA handles the transpose internally)
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte offset: (m + row) * stride + (n + col) * elt_size
    %elt_size = arith.constant 2 : index  // f16 = 2 bytes
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes (4xf16) via global_load_dwordx2
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return %result : !rt_B_f16
  }

  //===--------------------------------------------------------------------===//
  // Global memory store functions (Phase 2)
  //===--------------------------------------------------------------------===//

  // Store a 16x16 f16 tile to global memory from MFMA A fragment layout.
  // Each thread stores 4xf16 (8 bytes) via global_store_dwordx2.
  // Lane i stores to row (i % 16), cols [(i/16)*4, (i/16)*4 + 4).
  //
  // Parameters:
  //   %tile: !rt_A_f16 (= !vx2) tile data to store
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  func.func private @store_A_f16(%tile: !rt_A_f16, %ptr: !sx2, %m: index, %n: index, %stride: index) {
    // Get MFMA A index for this lane
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Compute byte offset: (m + row) * stride + (n + col) * elt_size
    %elt_size = arith.constant 2 : index  // f16 = 2 bytes
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Store 8 bytes (4xf16) via global_store_dwordx2
    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx2 data %tile addr %ptr offset d(%off_reg) + c(%c0) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // MFMA operations (Phase 3)
  //===--------------------------------------------------------------------===//

  // Perform a 16x16x16 matrix multiply-accumulate: D = A @ B^T + C
  // Uses v_mfma_f32_16x16x16_f16 instruction.
  //
  // The MFMA computes: D[16x16] = A[16x16] @ B[16x16]^T + C[16x16]
  // where A and B are f16, C and D are f32.
  //
  // Parameters:
  //   %A: !rt_A_f16 - A operand (16x16 f16 tile)
  //   %B: !rt_B_f16 - B operand (16x16 f16 tile, transposed internally)
  //   %C: !rt_C_f32 - C accumulator (16x16 f32 tile)
  //
  // Returns: !rt_C_f32 - D result (16x16 f32 tile)
  func.func private @mfma_f32_16x16x16_f16(%A: !rt_A_f16, %B: !rt_B_f16, %C: !rt_C_f32) -> !rt_C_f32 {
    %dst = func.call @alloc_vgprx4() : () -> !vx4
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %A, %B, %C
        : !vx2, !vx2, !vx4 -> !vx4
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory store functions for C tile (Phase 4)
  //===--------------------------------------------------------------------===//

  // Store a 16x16 f32 C tile to global memory from MFMA C fragment layout.
  // Each thread holds 4xf32 at 4 consecutive rows in the same column.
  // Lane i stores to rows [(i/16)*4, (i/16)*4+4), column (i % 16).
  //
  // Since these 4 values are non-contiguous in row-major memory (separated by
  // stride), we perform 4 separate global_store_dword operations.
  //
  // Parameters:
  //   %tile: !rt_C_f32 (= !vx4) tile data to store
  //   %ptr: base pointer to the matrix
  //   %m: row position of tile in elements
  //   %n: column position of tile in elements
  //   %stride: row stride in bytes
  func.func private @store_C_f32(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) {
    // Get MFMA C index for this lane.
    // mfma_index_C returns (i, j) where:
    //   i = lane_id % 16  -> column position in C matrix
    //   j = (lane_id / 16) * 4 -> base row position for 4-element group
    //
    // C fragment layout: Lane i holds C[(i/16)*4 : (i/16)*4+4, i%16]
    //   - 4 consecutive rows at the same column
    %mfma_idx = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Element size for f32
    %elt_size = arith.constant 4 : index  // f32 = 4 bytes
    %c0_i32 = arith.constant 0 : i32

    // Split the !vx4 into 4 individual f32 values
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %tile : !vx4

    // Store each of the 4 f32 values at consecutive row positions, same column
    // Position: (m + row_base + offset, n + col)

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

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }
}
