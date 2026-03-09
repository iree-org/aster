// Kittens compute primitives for 32x32 f16 MFMA operations.
// Provides MFMA wrapper and multi-MFMA chaining from LDS read futures.
// Accumulators use AGPRs: on gfx942 v_mfma_f32_32x32x8_f16 accepts AGPR
// for both C (input) and VDST (output).

// Register types
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax16 = !amdgcn.agpr<[? + 16]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax16

// Future/token types
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

// Buffer type aliases
!lds_rfut_buf = memref<?x!future_lds_read>

amdgcn.library @kittens_compute_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From futures.mlir
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // MFMA operation (AGPR accumulator)
  //===--------------------------------------------------------------------===//

  // D[32x32, agpr] = A[32x8, vgpr] @ B[32x8, vgpr]^T + C[32x32, agpr]
  func.func private @mfma_f32_32x32x8_f16(%A: !rt_A_f16, %B: !rt_B_f16, %C: !rt_C_f32) -> !rt_C_f32 {
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x8_f16> %C, %A, %B, %C
        : !vx2, !vx2, !ax16 -> !ax16
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Compute (4 MFMAs from LDS read futures)
  //===--------------------------------------------------------------------===//

  // Chain 4 MFMAs: extract values from LDS read futures and accumulate in AGPRs.
  // Takes memref<?x!future_lds_read> for A (4 entries) and B (4 entries).
  func.func private @compute_mfmas_32x32(
      %a_buf: !lds_rfut_buf, %b_buf: !lds_rfut_buf, %acc: !rt_C_f32
  ) -> !rt_C_f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %acc_buf = memref.alloca(%c1) : memref<?x!rt_C_f32>
    memref.store %acc, %acc_buf[%c0] : memref<?x!rt_C_f32>

    scf.for %k = %c0 to %c4 step %c1 {
      %a_fut = memref.load %a_buf[%k] : !lds_rfut_buf
      %b_fut = memref.load %b_buf[%k] : !lds_rfut_buf
      %A = func.call @get_lds_read_value_vx2(%a_fut) : (!future_lds_read) -> !vx2
      %B = func.call @get_lds_read_value_vx2(%b_fut) : (!future_lds_read) -> !vx2
      %c = memref.load %acc_buf[%c0] : memref<?x!rt_C_f32>
      %c_new = func.call @mfma_f32_32x32x8_f16(%A, %B, %c)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %acc_buf[%c0] : memref<?x!rt_C_f32>
    } {aster.constexpr}

    %result = memref.load %acc_buf[%c0] : memref<?x!rt_C_f32>
    return %result : !rt_C_f32
  }
}
