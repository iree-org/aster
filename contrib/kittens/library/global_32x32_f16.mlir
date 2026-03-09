// Kittens 32x32_f16 tile abstractions for global load/store.
// Uses ptr.ptr_add for address computation, letting aster-optimize-ptr-add
// decompose offsets into const/uniform/dynamic components for optimal codegen.
// Accumulators (C tiles) use AGPRs: on gfx942 MFMAs write directly to AGPRs
// and global_store_dword can read directly from AGPRs.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!s   = !amdgcn.sgpr
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!a   = !amdgcn.agpr
!ax16 = !amdgcn.agpr<[? + 16]>

// Ptr type for generic-space pointers (64-bit on AMDGCN)
!gptr = !ptr.ptr<#ptr.generic_space>

// Token types for async memory operations
!write_token = !amdgcn.write_token<flat>
!wtok_buf = memref<?x!write_token>

// Future types for global loads
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!gfut_buf = memref<?x!future_global_read>

// Descriptor types
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>

// Kittens register tile type aliases for 32x32x8 MFMA (AGPR accumulators)
!rt_C_f32 = !ax16

amdgcn.library @kittens_global_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_agprx16(i32) -> !ax16
  // From indexing.mlir (non-ptr versions still needed for MFMA indexing)
  func.func private @mfma_index_C_32x32xf32() -> !index_pair
  func.func private @mfma_c_row_32x32xf32(index, index) -> index
  func.func private @thread_tile_pos_32x32() -> (index, index)
  // From indexing_ptr.mlir
  func.func private @matrix_offset_idx(!index_descriptor_2d) -> index
  func.func private @tile_thread_offset_idx(!index_descriptor_2d, index) -> index
  func.func private @global_addr_from_offset(!gptr, index) -> !vx2

  //===--------------------------------------------------------------------===//
  // AGPR Tile initialization
  //===--------------------------------------------------------------------===//

  func.func private @zero_C_32x32() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_agprx16(%c0) : (i32) -> !ax16
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global Load (32x32 tile, ptr-based addressing)
  //===--------------------------------------------------------------------===//

  // Issue 4 global loads for a 32x32 f16 tile using ptr.ptr_add addressing.
  // The offset stays as index until the final ptr.ptr_add, which takes i32.
  // aster-optimize-ptr-add decomposes the offset into const/uniform/dynamic,
  // then aster-codegen lowers to amdgcn.ptr_add with proper register classes.
  func.func private @load_global_tile_32x32_f16(
      %ptr: !sx2, %m: index, %k_base: index, %stride: index
  ) -> !gfut_buf {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // Thread-level offset components (per-lane from lane_id)
    %d_desc = aster_utils.struct_create(%row_in_group, %col, %stride, %elt_size)
        : (index, index, index, index) -> !index_descriptor_2d
    %d_off = func.call @matrix_offset_idx(%d_desc) : (!index_descriptor_2d) -> index

    // Bridge from register-level SGPR pair to ptr dialect type
    %gptr = lsir.from_reg %ptr : !sx2 -> !gptr
    %buf = memref.alloca(%c4) : !gfut_buf
    scf.for %g = %c0 to %c4 step %c1 {
      // source address calculation
      %tile_row = affine.apply affine_map<(g)[m] -> (m + g * 8)>(%g)[%m]
      %u_desc = aster_utils.struct_create(%tile_row, %k_base, %stride, %elt_size)
          : (index, index, index, index) -> !index_descriptor_2d
      %total_off = func.call @tile_thread_offset_idx(%u_desc, %d_off)
          : (!index_descriptor_2d, index) -> index

      %addr = func.call @global_addr_from_offset(%gptr, %total_off)
          : (!gptr, index) -> !vx2

      // load
      %tmp = lsir.alloca : !vx2
      %loaded, %tok = amdgcn.load global_load_dwordx2 dest %tmp addr %addr
          : dps(!vx2) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
      %val = aster_utils.to_any %loaded : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
      memref.store %f, %buf[%g] : !gfut_buf
    } {aster.constexpr}

    return %buf : !gfut_buf
  }

  //===--------------------------------------------------------------------===//
  // Global Store (C tile 32x32 f32 from AGPRs, ptr-based addressing)
  //===--------------------------------------------------------------------===//

  // Store a 32x32 f32 C tile from AGPRs using ptr.ptr_add addressing.
  // On gfx942, global_store_dword can read directly from AGPRs.
  func.func private @store_C_32x32_f32(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) -> !wtok_buf {
    %mfma_idx = func.call @mfma_index_C_32x32xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

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

    // Split !ax16 into 16 individual AGPRs and pack into buffer for iteration.
    %r:16 = amdgcn.split_register_range %tile : !ax16
    %reg_buf = memref.alloca(%c16) : memref<?x!aster_utils.any>
    %a0  = aster_utils.to_any %r#0  : !a
    %a1  = aster_utils.to_any %r#1  : !a
    %a2  = aster_utils.to_any %r#2  : !a
    %a3  = aster_utils.to_any %r#3  : !a
    %a4  = aster_utils.to_any %r#4  : !a
    %a5  = aster_utils.to_any %r#5  : !a
    %a6  = aster_utils.to_any %r#6  : !a
    %a7  = aster_utils.to_any %r#7  : !a
    %a8  = aster_utils.to_any %r#8  : !a
    %a9  = aster_utils.to_any %r#9  : !a
    %a10 = aster_utils.to_any %r#10 : !a
    %a11 = aster_utils.to_any %r#11 : !a
    %a12 = aster_utils.to_any %r#12 : !a
    %a13 = aster_utils.to_any %r#13 : !a
    %a14 = aster_utils.to_any %r#14 : !a
    %a15 = aster_utils.to_any %r#15 : !a
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

    // Thread-level offset: row_base * stride + col * elt_size (per-lane)
    %elt_size = arith.constant 4 : index
    %d_desc = aster_utils.struct_create(%row_base, %col, %stride, %elt_size)
        : (index, index, index, index) -> !index_descriptor_2d
    %d_off = func.call @matrix_offset_idx(%d_desc) : (!index_descriptor_2d) -> index

    // Bridge from register-level SGPR pair to ptr dialect type
    %gptr = lsir.from_reg %ptr : !sx2 -> !gptr
    %tok_buf = memref.alloca(%c16) : memref<?x!write_token>
    scf.for %i = %c0 to %c16 step %c1 {
      %any_reg = memref.load %reg_buf[%i] : memref<?x!aster_utils.any>
      %reg = aster_utils.from_any %any_reg : !a

      // target address calculation
      %reg_row_const = func.call @mfma_c_row_32x32xf32(%c0, %i) : (index, index) -> index
      %tile_row = affine.apply affine_map<()[m, rrc] -> (m + rrc)>()[%m, %reg_row_const]
      %u_desc = aster_utils.struct_create(%tile_row, %n, %stride, %elt_size)
          : (index, index, index, index) -> !index_descriptor_2d
      %total_off = func.call @tile_thread_offset_idx(%u_desc, %d_off)
          : (!index_descriptor_2d, index) -> index

      %addr = func.call @global_addr_from_offset(%gptr, %total_off)
          : (!gptr, index) -> !vx2

      // store from AGPR (gfx942 reads AGPRs directly for global_store)
      %tok = amdgcn.store global_store_dword data %reg addr %addr
          : ins(!amdgcn.agpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
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
