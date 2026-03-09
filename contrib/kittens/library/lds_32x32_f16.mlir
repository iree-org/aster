// Kittens LDS primitives for 32x32 f16 tiles (feeding 32x32x8 MFMA).
//
// Uses lsir.alloca for load destinations. Address computation stays as index
// until the final lsir.to_reg at the load/store site.
//
// LDS addressing is flat (no base pointer), so amdgcn.ptr_add does not apply.
// The entire address is a byte offset in VGPR computed via XOR swizzle.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax16 = !amdgcn.agpr<[? + 16]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax16

// Future/token types
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Buffer type aliases for memref-based function signatures
!gfut_buf = memref<?x!future_global_read>
!lds_wtok_buf = memref<?x!future_lds_write>
!lds_rfut_buf = memref<?x!future_lds_read>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.library @kittens_lds_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @mfma_index_A_32x32xf16() -> !index_pair
  func.func private @mfma_index_B_32x32xf16() -> !index_pair
  func.func private @thread_tile_pos_32x32() -> (index, index)
  func.func private @lds_xor_swizzled_addr_32x32(index, index, index) -> index
  // From indexing_ptr.mlir
  func.func private @index_to_vgpr_i32(index) -> !v
  // From futures.mlir
  func.func private @get_global_load_value_vx2(!future_global_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // LDS Store (32x32 tile, XOR-swizzled row-major, stride = 64 bytes/row)
  //===--------------------------------------------------------------------===//

  // Store global load futures to LDS as a 32x32 XOR-swizzled tile.
  // Takes memref<?x!future_global_read> (4 entries, one per row group).
  // Returns memref<?x!future_lds_write> (4 write tokens).
  func.func private @store_global_tile_to_lds_32x32_f16(
      %lds_base: index, %gf_buf: !gfut_buf
  ) -> !lds_wtok_buf {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %byte_in_row = affine.apply affine_map<(c) -> (c * 2)>(%col)

    %tok_buf = memref.alloca(%c4) : !lds_wtok_buf
    scf.for %g = %c0 to %c4 step %c1 {
      %gf = memref.load %gf_buf[%g] : !gfut_buf
      %loaded = func.call @get_global_load_value_vx2(%gf) : (!future_global_read) -> !vx2
      %row = affine.apply affine_map<(g)[rig] -> (rig + g * 8)>(%g)[%row_in_group]
      %addr_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte_in_row)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%addr_idx) : (index) -> !v
      %tok = amdgcn.store ds_write_b64 data %loaded addr %addr offset c(%c0_i32)
          : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
      memref.store %tok, %tok_buf[%g] : !lds_wtok_buf
    } {aster.constexpr}

    return %tok_buf : !lds_wtok_buf
  }

  // Wait for all LDS write tokens in a buffer.
  func.func private @wait_lds_writes_32x32(%tok_buf: !lds_wtok_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %tok = memref.load %tok_buf[%i] : !lds_wtok_buf
      amdgcn.wait deps %tok : !future_lds_write
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // LDS Read for MFMA (32x8 fragments from 32x32 XOR-swizzled LDS)
  //===--------------------------------------------------------------------===//

  // Read 4 MFMA A fragments from a 32x32 XOR-swizzled tile in LDS.
  // Sub-tile k (0..3) reads K cols k*8..k*8+7.
  // byte_in_row for sub-tile k = k*16 + mfma_col*2.
  // Returns memref<?x!future_lds_read> with 4 entries.
  func.func private @load_lds_A_32x32_f16(%lds_base: index) -> !lds_rfut_buf {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca(%c4) : !lds_rfut_buf
    scf.for %k = %c0 to %c4 step %c1 {
      %byte = affine.apply affine_map<(k, c) -> (k * 16 + c * 2)>(%k, %col)
      %off_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%off_idx) : (index) -> !v
      %dst = lsir.alloca : !vx2
      %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
          : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
      %val = aster_utils.to_any %result : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
      memref.store %f, %buf[%k] : !lds_rfut_buf
    } {aster.constexpr}

    return %buf : !lds_rfut_buf
  }

  // Read 4 MFMA B fragments from a 32x32 XOR-swizzled tile in LDS.
  // Same swizzle formula as A, but using B indexing (reversed i/j extraction).
  // Returns memref<?x!future_lds_read> with 4 entries.
  func.func private @load_lds_B_32x32_f16(%lds_base: index) -> !lds_rfut_buf {
    %mfma_idx = func.call @mfma_index_B_32x32xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca(%c4) : !lds_rfut_buf
    scf.for %k = %c0 to %c4 step %c1 {
      %byte = affine.apply affine_map<(k, c) -> (k * 16 + c * 2)>(%k, %col)
      %off_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%off_idx) : (index) -> !v
      %dst = lsir.alloca : !vx2
      %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
          : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
      %val = aster_utils.to_any %result : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
      memref.store %f, %buf[%k] : !lds_rfut_buf
    } {aster.constexpr}

    return %buf : !lds_rfut_buf
  }

}
