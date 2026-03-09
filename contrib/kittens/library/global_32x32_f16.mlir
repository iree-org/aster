// Kittens-style 32x32_f16 tile abstractions for global load/store feeding into
// MFMA 32x32x8 operations.
// Provides high-level primitives for register tiles used in GEMM kernels.

// Register types from common library
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx16 = !amdgcn.vgpr<[? + 16]>

// Token types for async memory operations
!write_token = !amdgcn.write_token<flat>
!wtok_buf = memref<?x!write_token>

// Future types for global loads
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!gfut_buf = memref<?x!future_global_read>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

// Kittens register tile type aliases for 32x32x8 MFMA
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx16

amdgcn.library @kittens_global_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_vgprx16(i32) -> !vx16
  func.func private @alloc_vgprx2() -> !vx2
  // From indexing.mlir
  func.func private @mfma_index_C_32x32xf32() -> !index_pair
  func.func private @mfma_c_row_32x32xf32(index, index) -> index
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @thread_tile_pos_32x32() -> (index, index)

  //===--------------------------------------------------------------------===//
  // Tile initialization
  //===--------------------------------------------------------------------===//

  // Initialize a 32x32 f32 accumulator tile to zero.
  func.func private @zero_C_32x32() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_vgprx16(%c0) : (i32) -> !vx16
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global Load (32x32 tile, 4 coalesced row-group loads)
  //===--------------------------------------------------------------------===//

  // Issue 4 global loads for a 32x32 f16 tile with coalesced access.
  // Each load covers 8 rows (8 threads/row * 4 f16/thread = full 32-element row).
  // Returns memref<?x!future_global_read> with 4 entries (row groups 0-7, 8-15, 16-23, 24-31).
  // TODO: Generalize beyond %c4 and pass index as a function arg for reuse.
  func.func private @load_global_tile_32x32_f16(
      %ptr: !sx2, %m: index, %k_base: index, %stride: index
  ) -> !gfut_buf {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca(%c4) : !gfut_buf
    scf.for %g = %c0 to %c4 step %c1 {
      %row = affine.apply affine_map<(g)[rig] -> (rig + g * 8)>(%g)[%row_in_group]
      %desc = aster_utils.struct_create(%m, %k_base, %row, %col, %stride, %elt_size)
          : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
      %off = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v
      %tmp = func.call @alloc_vgprx2() : () -> !vx2
      %loaded, %tok = amdgcn.load global_load_dwordx2 dest %tmp addr %ptr
          offset d(%off) + c(%c0_i32) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
      %val = aster_utils.to_any %loaded : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
      memref.store %f, %buf[%g] : !gfut_buf
    } {aster.constexpr}

    return %buf : !gfut_buf
  }

  //===--------------------------------------------------------------------===//
  // Global memory store for C tile (32x32 f32)
  //===--------------------------------------------------------------------===//

  // Store a 32x32 f32 C tile to global memory from MFMA C fragment layout.
  // Each thread holds 16xf32 across 16 VGPRs.
  //
  // split_register_range produces 16 separate SSA values that must be stored
  // to a buffer before they can be iterated in a constexpr loop.
  // After constexpr expansion + SROA + mem2reg, the buffers disappear.
  func.func private @store_C_32x32_f32(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) -> !wtok_buf {
    %mfma_idx = func.call @mfma_index_C_32x32xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    // Note: this manual unrolling is unfortunately necessary because of the !vx16 data type for %tile: !rt_C_f32.
    // TODO: revisit whether we can pass memref<?x!v>
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
    %c3  = arith.constant 3 : index
    %c4  = arith.constant 4 : index
    %c5  = arith.constant 5 : index
    %c6  = arith.constant 6 : index
    %c7  = arith.constant 7 : index
    %c8  = arith.constant 8 : index
    %c9  = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index

    // Split !vx16 into 16 individual VGPRs and pack into buffer for iteration.
    %r:16 = amdgcn.split_register_range %tile : !vx16
    %reg_buf = memref.alloca(%c16) : memref<?x!aster_utils.any>
    %a0  = aster_utils.to_any %r#0  : !v
    %a1  = aster_utils.to_any %r#1  : !v
    %a2  = aster_utils.to_any %r#2  : !v
    %a3  = aster_utils.to_any %r#3  : !v
    %a4  = aster_utils.to_any %r#4  : !v
    %a5  = aster_utils.to_any %r#5  : !v
    %a6  = aster_utils.to_any %r#6  : !v
    %a7  = aster_utils.to_any %r#7  : !v
    %a8  = aster_utils.to_any %r#8  : !v
    %a9  = aster_utils.to_any %r#9  : !v
    %a10 = aster_utils.to_any %r#10 : !v
    %a11 = aster_utils.to_any %r#11 : !v
    %a12 = aster_utils.to_any %r#12 : !v
    %a13 = aster_utils.to_any %r#13 : !v
    %a14 = aster_utils.to_any %r#14 : !v
    %a15 = aster_utils.to_any %r#15 : !v
    memref.store %a0,  %reg_buf[%c0]  : memref<?x!aster_utils.any>
    memref.store %a1,  %reg_buf[%c1]  : memref<?x!aster_utils.any>
    memref.store %a2,  %reg_buf[%c2]  : memref<?x!aster_utils.any>
    memref.store %a3,  %reg_buf[%c3]  : memref<?x!aster_utils.any>
    memref.store %a4,  %reg_buf[%c4]  : memref<?x!aster_utils.any>
    memref.store %a5,  %reg_buf[%c5]  : memref<?x!aster_utils.any>
    memref.store %a6,  %reg_buf[%c6]  : memref<?x!aster_utils.any>
    memref.store %a7,  %reg_buf[%c7]  : memref<?x!aster_utils.any>
    memref.store %a8,  %reg_buf[%c8]  : memref<?x!aster_utils.any>
    memref.store %a9,  %reg_buf[%c9]  : memref<?x!aster_utils.any>
    memref.store %a10, %reg_buf[%c10] : memref<?x!aster_utils.any>
    memref.store %a11, %reg_buf[%c11] : memref<?x!aster_utils.any>
    memref.store %a12, %reg_buf[%c12] : memref<?x!aster_utils.any>
    memref.store %a13, %reg_buf[%c13] : memref<?x!aster_utils.any>
    memref.store %a14, %reg_buf[%c14] : memref<?x!aster_utils.any>
    memref.store %a15, %reg_buf[%c15] : memref<?x!aster_utils.any>

    // Iterate over all 16 registers, compute row from fragment layout, store to global.
    %elt_size = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %tok_buf = memref.alloca(%c16) : memref<?x!write_token>
    scf.for %i = %c0 to %c16 step %c1 {
      %any_reg = memref.load %reg_buf[%i] : memref<?x!aster_utils.any>
      %reg = aster_utils.from_any %any_reg : !v
      %row = func.call @mfma_c_row_32x32xf32(%row_base, %i) : (index, index) -> index
      %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size)
          : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
      %off = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v
      %tok = amdgcn.store global_store_dword data %reg addr %ptr offset d(%off) + c(%c0_i32)
          : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
      memref.store %tok, %tok_buf[%i] : memref<?x!write_token>
    } {aster.constexpr}

    return %tok_buf : !wtok_buf
  }

  // Wait for all global store tokens from store_C_32x32_f32.
  func.func private @wait_global_writes_32x32(%tok_buf: !wtok_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %tok = memref.load %tok_buf[%i] : !wtok_buf
      amdgcn.wait deps %tok : !write_token
    } {aster.constexpr}
    return
  }
}
