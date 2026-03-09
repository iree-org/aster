// Kittens LDS primitives for 32x32 f16 tiles (feeding 32x32x8 MFMA).
//
// Uses lsir.alloca for load destinations. Address computation stays as index
// until the final lsir.to_reg at the load/store site.
//
// LDS addressing is flat (no base pointer), so amdgcn.ptr_add does not apply.
// The entire address is a byte offset in VGPR computed via XOR swizzle.
//
// Library functions are split into Phase 1 (VALU) and Phase 2 (DS) halves
// to enable kernel-level FU type batching. Combined wrappers are provided
// for callers that don't need the split (e.g. unit tests).

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
  // LDS Store - Split API (32x32 tile, XOR-swizzled)
  //===--------------------------------------------------------------------===//

  // Phase 1 (VALU): Extract data from global futures + compute LDS addresses.
  // Returns (data_buf[4], addr_buf[4]).
  func.func private @prepare_lds_write_32x32_f16(
      %lds_base: index, %gf_buf: !gfut_buf
  ) -> (memref<?x!vx2>, memref<?x!v>) {
    %row_in_group, %col = func.call @thread_tile_pos_32x32() : () -> (index, index)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %byte_in_row = affine.apply affine_map<(c) -> (c * 2)>(%col)

    %data_buf = memref.alloca(%c4) : memref<?x!vx2>
    %addr_buf = memref.alloca(%c4) : memref<?x!v>
    scf.for %g = %c0 to %c4 step %c1 {
      %gf = memref.load %gf_buf[%g] : !gfut_buf
      %loaded = func.call @get_global_load_value_vx2(%gf) : (!future_global_read) -> !vx2
      memref.store %loaded, %data_buf[%g] : memref<?x!vx2>
      %row = affine.apply affine_map<(g)[rig] -> (rig + g * 8)>(%g)[%row_in_group]
      %addr_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte_in_row)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%addr_idx) : (index) -> !v
      memref.store %addr, %addr_buf[%g] : memref<?x!v>
    } {aster.constexpr}

    return %data_buf, %addr_buf : memref<?x!vx2>, memref<?x!v>
  }

  // Phase 2 (DS): Issue 4 LDS writes from pre-computed data and addresses.
  func.func private @issue_lds_writes_32x32_f16(
      %data_buf: memref<?x!vx2>, %addr_buf: memref<?x!v>
  ) -> !lds_wtok_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %tok_buf = memref.alloca(%c4) : !lds_wtok_buf
    scf.for %g = %c0 to %c4 step %c1 {
      %loaded = memref.load %data_buf[%g] : memref<?x!vx2>
      %addr = memref.load %addr_buf[%g] : memref<?x!v>
      %tok = amdgcn.store ds_write_b64 data %loaded addr %addr offset c(%c0_i32)
          : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
      memref.store %tok, %tok_buf[%g] : !lds_wtok_buf
    } {aster.constexpr}

    return %tok_buf : !lds_wtok_buf
  }

  // Combined wrapper (calls Phase 1 + Phase 2). Used by unit tests.
  func.func private @store_global_tile_to_lds_32x32_f16(
      %lds_base: index, %gf_buf: !gfut_buf
  ) -> !lds_wtok_buf {
    %data_buf, %addr_buf = func.call @prepare_lds_write_32x32_f16(
        %lds_base, %gf_buf)
        : (index, !gfut_buf) -> (memref<?x!vx2>, memref<?x!v>)
    %tok_buf = func.call @issue_lds_writes_32x32_f16(%data_buf, %addr_buf)
        : (memref<?x!vx2>, memref<?x!v>) -> !lds_wtok_buf
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
  // LDS Read A - Split API (32x8 MFMA fragments from 32x32 XOR-swizzled LDS)
  //===--------------------------------------------------------------------===//

  // Phase 1 (VALU): Compute 4 LDS read addresses + allocate destinations for A.
  // Returns (addr_buf[4], dst_buf[4]).
  func.func private @compute_lds_A_addrs_32x32_f16(
      %lds_base: index
  ) -> (memref<?x!v>, memref<?x!vx2>) {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %addr_buf = memref.alloca(%c4) : memref<?x!v>
    %dst_buf = memref.alloca(%c4) : memref<?x!vx2>
    scf.for %k = %c0 to %c4 step %c1 {
      %byte = affine.apply affine_map<(k, c) -> (k * 16 + c * 2)>(%k, %col)
      %off_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%off_idx) : (index) -> !v
      memref.store %addr, %addr_buf[%k] : memref<?x!v>
      %dst = lsir.alloca : !vx2
      memref.store %dst, %dst_buf[%k] : memref<?x!vx2>
    } {aster.constexpr}

    return %addr_buf, %dst_buf : memref<?x!v>, memref<?x!vx2>
  }

  // Phase 2 (DS): Issue 4 LDS reads for A from pre-computed addresses.
  func.func private @issue_lds_reads_A_32x32_f16(
      %addr_buf: memref<?x!v>, %dst_buf: memref<?x!vx2>
  ) -> !lds_rfut_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca(%c4) : !lds_rfut_buf
    scf.for %k = %c0 to %c4 step %c1 {
      %addr = memref.load %addr_buf[%k] : memref<?x!v>
      %dst = memref.load %dst_buf[%k] : memref<?x!vx2>
      %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
          : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
      %val = aster_utils.to_any %result : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
      memref.store %f, %buf[%k] : !lds_rfut_buf
    } {aster.constexpr}

    return %buf : !lds_rfut_buf
  }

  // Combined wrapper (calls Phase 1 + Phase 2). Used by unit tests.
  func.func private @load_lds_A_32x32_f16(%lds_base: index) -> !lds_rfut_buf {
    %addr_buf, %dst_buf = func.call @compute_lds_A_addrs_32x32_f16(%lds_base)
        : (index) -> (memref<?x!v>, memref<?x!vx2>)
    %buf = func.call @issue_lds_reads_A_32x32_f16(%addr_buf, %dst_buf)
        : (memref<?x!v>, memref<?x!vx2>) -> !lds_rfut_buf
    return %buf : !lds_rfut_buf
  }

  //===--------------------------------------------------------------------===//
  // LDS Read B - Split API (32x8 MFMA fragments from 32x32 XOR-swizzled LDS)
  //===--------------------------------------------------------------------===//

  // Phase 1 (VALU): Compute 4 LDS read addresses + allocate destinations for B.
  // Returns (addr_buf[4], dst_buf[4]).
  func.func private @compute_lds_B_addrs_32x32_f16(
      %lds_base: index
  ) -> (memref<?x!v>, memref<?x!vx2>) {
    %mfma_idx = func.call @mfma_index_B_32x32xf16() : () -> !index_pair
    %col, %row = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %addr_buf = memref.alloca(%c4) : memref<?x!v>
    %dst_buf = memref.alloca(%c4) : memref<?x!vx2>
    scf.for %k = %c0 to %c4 step %c1 {
      %byte = affine.apply affine_map<(k, c) -> (k * 16 + c * 2)>(%k, %col)
      %off_idx = func.call @lds_xor_swizzled_addr_32x32(%lds_base, %row, %byte)
          : (index, index, index) -> index
      %addr = func.call @index_to_vgpr_i32(%off_idx) : (index) -> !v
      memref.store %addr, %addr_buf[%k] : memref<?x!v>
      %dst = lsir.alloca : !vx2
      memref.store %dst, %dst_buf[%k] : memref<?x!vx2>
    } {aster.constexpr}

    return %addr_buf, %dst_buf : memref<?x!v>, memref<?x!vx2>
  }

  // Phase 2 (DS): Issue 4 LDS reads for B from pre-computed addresses.
  func.func private @issue_lds_reads_B_32x32_f16(
      %addr_buf: memref<?x!v>, %dst_buf: memref<?x!vx2>
  ) -> !lds_rfut_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    %buf = memref.alloca(%c4) : !lds_rfut_buf
    scf.for %k = %c0 to %c4 step %c1 {
      %addr = memref.load %addr_buf[%k] : memref<?x!v>
      %dst = memref.load %dst_buf[%k] : memref<?x!vx2>
      %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %addr offset c(%c0_i32)
          : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
      %val = aster_utils.to_any %result : !vx2
      %f = aster_utils.struct_create(%val, %tok)
          : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read
      memref.store %f, %buf[%k] : !lds_rfut_buf
    } {aster.constexpr}

    return %buf : !lds_rfut_buf
  }

  // Combined wrapper (calls Phase 1 + Phase 2). Used by unit tests.
  func.func private @load_lds_B_32x32_f16(%lds_base: index) -> !lds_rfut_buf {
    %addr_buf, %dst_buf = func.call @compute_lds_B_addrs_32x32_f16(%lds_base)
        : (index) -> (memref<?x!v>, memref<?x!vx2>)
    %buf = func.call @issue_lds_reads_B_32x32_f16(%addr_buf, %dst_buf)
        : (memref<?x!v>, memref<?x!vx2>) -> !lds_rfut_buf
    return %buf : !lds_rfut_buf
  }

}
