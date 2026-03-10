// Kittens compute primitives for 16x16x16 f16 MFMA with AGPR accumulators.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!a   = !amdgcn.agpr
!ax4 = !amdgcn.agpr<[? + 4]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.library @kittens_compute_16x16_f16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_agprx4(i32) -> !ax4
  // From indexing.mlir
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  func.func private @mfma_c_16x16xf32_byte_offset(index, index, index, index, index, index, index) -> index
  // From indexing_ptr.mlir
  func.func private @global_addr_from_offset(!sx2, index) -> !vx2

  //===--------------------------------------------------------------------===//
  // Accumulator init (AGPR)
  //===--------------------------------------------------------------------===//

  // Initialize a 16x16 f32 accumulator tile to zero in AGPRs.
  func.func private @zero_C() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_agprx4(%c0) : (i32) -> !ax4
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // MFMA operation (AGPR accumulator)
  //===--------------------------------------------------------------------===//

  // D[16x16, agpr] = A[16x16, vgpr] @ B[16x16, vgpr]^T + C[16x16, agpr]
  func.func private @mfma_f32_16x16x16_f16(%A: !rt_A_f16, %B: !rt_B_f16, %C: !rt_C_f32) -> !rt_C_f32 {
    // Accumulator-in-place: reuse %C as DPS destination for loop compatibility.
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %C, %A, %B, %C
        : !vx2, !vx2, !ax4 -> !ax4
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Fire-and-forget C tile store (AGPR, ptr-based addressing)
  //===--------------------------------------------------------------------===//

  // Store a 16x16 f32 C tile from AGPRs to global memory in MFMA C fragment layout.
  // Each thread holds 4xf32 at 4 consecutive rows in the same column.
  // Fire-and-forget: no tokens returned. s_endpgm drains all outstanding stores.
  func.func private @store_global_C_mfma_f32_16x16x16_f16(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) {
    // Note: hardcoded element size is related to layout and producing mfma variant.
    // Cannot just be generalized without more effort.
    %elt_size = arith.constant 4 : index

    // C fragment layout: Lane l holds C[(l/16)*4 : (l/16)*4+4, l%16]
    %mfma_idx = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Split !ax4 into 4 individual AGPRs and collect into a buffer for constexpr loop access.
    // Note: this could be further folded into loops and be made more generic if
    // we had a amdgcn.extract_register %tile, %i to turn static into dynamic +
    // late constexpr unrolling.
    %a0, %a1, %a2, %a3 = amdgcn.split_register_range %tile : !ax4
    %agpr_buf = memref.alloca(%c4) : memref<?x!a>
    memref.store %a0, %agpr_buf[%c0] : memref<?x!a>
    memref.store %a1, %agpr_buf[%c1] : memref<?x!a>
    memref.store %a2, %agpr_buf[%c2] : memref<?x!a>
    memref.store %a3, %agpr_buf[%c3] : memref<?x!a>

    // Compute byte offsets and addresses for 4 consecutive rows.
    %addr_buf = memref.alloca(%c4) : memref<?x!vx2>
    scf.for %i = %c0 to %c4 step %c1 {
      %off = func.call @mfma_c_16x16xf32_byte_offset(%m, %n, %row_base, %col, %stride, %elt_size, %i)
          : (index, index, index, index, index, index, index) -> index
      %addr = func.call @global_addr_from_offset(%ptr, %off) : (!sx2, index) -> !vx2
      memref.store %addr, %addr_buf[%i] : memref<?x!vx2>
    } {aster.constexpr}

    // Fire-and-forget stores from AGPRs (gfx942 reads AGPRs directly).
    scf.for %i = %c0 to %c4 step %c1 {
      %addr = memref.load %addr_buf[%i] : memref<?x!vx2>
      %agpr = memref.load %agpr_buf[%i] : memref<?x!a>
      amdgcn.store global_store_dword data %agpr addr %addr
          : ins(!a, !vx2) -> !amdgcn.write_token<flat>
    } {aster.constexpr}

    return
  }
}
