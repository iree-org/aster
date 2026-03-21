// Kittens compute primitives for 32x32x8 f16 MFMA with AGPR accumulators.
//
// v_mfma_f32_32x32x8_f16: 8192 ops/instruction, 16 AGPRs per accumulator tile.
//   A fragment: lane l holds row l%32, cols [(l/32)*4 : (l/32)*4+4] in f16.
//   B fragment: same physical layout as A (B is transposed).
//   C fragment: lane l holds col l%32, 16 rows in 4 groups of 4 rows spaced 8 apart.
//     row = row_base + 8*(reg_idx/4) + reg_idx%4, where row_base = (l/32)*4.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!a   = !amdgcn.agpr
!ax16 = !amdgcn.agpr<[? + 16]>

// Kittens register tile types for 32x32x8
!rt_A_f16_32 = !vx2
!rt_B_f16_32 = !vx2
!rt_C_f32_32 = !ax16

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.library @kittens_compute_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_agprx16(i32) -> !ax16
  // From indexing.mlir
  func.func private @mfma_index_C_32x32_f32() -> !index_pair
  func.func private @mfma_c_row_32x32_f32(index, index) -> index
  // From indexing_ptr.mlir
  func.func private @global_addr_from_offset(!sx2, index) -> !vx2

  //===--------------------------------------------------------------------===//
  // Accumulator init (AGPR)
  //===--------------------------------------------------------------------===//

  // Initialize a 32x32 f32 accumulator tile to zero in AGPRs.
  func.func private @zero_C_32x32() -> !rt_C_f32_32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_agprx16(%c0) : (i32) -> !ax16
    return %result : !rt_C_f32_32
  }

  //===--------------------------------------------------------------------===//
  // MFMA operation (AGPR accumulator)
  //===--------------------------------------------------------------------===//

  // D[32x32, agpr] = A[32x8, vgpr] @ B[32x8, vgpr]^T + C[32x32, agpr]
  // A: lane l holds A[l%32][(l/32)*4 : (l/32)*4+4] (4 f16 in 2 VGPRs).
  // B: same physical layout (transposed in semantics).
  // C: 16 AGPRs per lane, covering a 32x32 f32 output region.
  func.func private @mfma_f32_32x32x8_f16(%A: !rt_A_f16_32, %B: !rt_B_f16_32, %C: !rt_C_f32_32) -> !rt_C_f32_32 {
    // Accumulator-in-place: reuse %C as DPS destination for loop compatibility.
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x8_f16> %C, %A, %B, %C
        : !vx2, !vx2, !ax16 -> !ax16
    return %result : !rt_C_f32_32
  }

  //===--------------------------------------------------------------------===//
  // Fire-and-forget C tile store (AGPR, ptr-based addressing)
  //===--------------------------------------------------------------------===//

  // Store a 32x32 f32 C tile from AGPRs to global memory in MFMA C fragment layout.
  // C fragment (v_mfma_f32_32x32x8_f16):
  //   col = lane_id % 32
  //   row_base = (lane_id / 32) * 4  (0 for lanes 0-31, 4 for lanes 32-63)
  //   register r (0..15) -> row = row_base + 8*(r/4) + r%4
  //     r=0..3  : rows row_base+0,1,2,3
  //     r=4..7  : rows row_base+8,9,10,11
  //     r=8..11 : rows row_base+16,17,18,19
  //     r=12..15: rows row_base+24,25,26,27
  // Fire-and-forget: no tokens returned. s_endpgm drains all outstanding stores.
  func.func private @store_global_C_mfma_f32_32x32x8_f16(%tile: !rt_C_f32_32, %global_ptr: !aster_utils.any, %m: index, %n: index, %stride: index) {
    %global_ptr_sx2 = aster_utils.from_any %global_ptr : !sx2
    // Note: element size is 4 bytes (f32).
    %elt_size = arith.constant 4 : index

    // C fragment layout: col = lane%32, row_base = (lane/32)*4.
    %mfma_idx = func.call @mfma_index_C_32x32_f32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %c0  = arith.constant 0  : index
    %c1  = arith.constant 1  : index
    %c16 = arith.constant 16 : index

    // Split !ax16 into 16 individual AGPRs.
    %a0, %a1, %a2,  %a3,  %a4,  %a5,  %a6,  %a7,
    %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15 =
        amdgcn.split_register_range %tile : !ax16

    %agpr_buf = memref.alloca(%c16) : memref<?x!a>
    memref.store %a0,  %agpr_buf[%c0]  : memref<?x!a>
    %c2  = arith.constant 2  : index
    %c3  = arith.constant 3  : index
    %c4  = arith.constant 4  : index
    %c5  = arith.constant 5  : index
    %c6  = arith.constant 6  : index
    %c7  = arith.constant 7  : index
    %c8  = arith.constant 8  : index
    %c9  = arith.constant 9  : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    memref.store %a1,  %agpr_buf[%c1]  : memref<?x!a>
    memref.store %a2,  %agpr_buf[%c2]  : memref<?x!a>
    memref.store %a3,  %agpr_buf[%c3]  : memref<?x!a>
    memref.store %a4,  %agpr_buf[%c4]  : memref<?x!a>
    memref.store %a5,  %agpr_buf[%c5]  : memref<?x!a>
    memref.store %a6,  %agpr_buf[%c6]  : memref<?x!a>
    memref.store %a7,  %agpr_buf[%c7]  : memref<?x!a>
    memref.store %a8,  %agpr_buf[%c8]  : memref<?x!a>
    memref.store %a9,  %agpr_buf[%c9]  : memref<?x!a>
    memref.store %a10, %agpr_buf[%c10] : memref<?x!a>
    memref.store %a11, %agpr_buf[%c11] : memref<?x!a>
    memref.store %a12, %agpr_buf[%c12] : memref<?x!a>
    memref.store %a13, %agpr_buf[%c13] : memref<?x!a>
    memref.store %a14, %agpr_buf[%c14] : memref<?x!a>
    memref.store %a15, %agpr_buf[%c15] : memref<?x!a>

    // Compute byte offsets for all 16 registers: (m + row) * stride + (n + col) * elt_size.
    %addr_buf = memref.alloca(%c16) : memref<?x!vx2>
    scf.for %i = %c0 to %c16 step %c1 {
      %row = func.call @mfma_c_row_32x32_f32(%row_base, %i) : (index, index) -> index
      %off = affine.apply
          affine_map<()[m, row, n, col, s, e] -> ((m + row) * s + (n + col) * e)>
          ()[%m, %row, %n, %col, %stride, %elt_size]
      %addr = func.call @global_addr_from_offset(%global_ptr_sx2, %off) : (!sx2, index) -> !vx2
      memref.store %addr, %addr_buf[%i] : memref<?x!vx2>
    } {aster.constexpr}

    // Fire-and-forget stores from AGPRs (gfx942 reads AGPRs directly).
    scf.for %i = %c0 to %c16 step %c1 {
      %addr = memref.load %addr_buf[%i] : memref<?x!vx2>
      %agpr = memref.load %agpr_buf[%i] : memref<?x!a>
      amdgcn.store global_store_dword data %agpr addr %addr
          : ins(!a, !vx2) -> !amdgcn.write_token<flat>
    } {aster.constexpr}

    return
  }
}
