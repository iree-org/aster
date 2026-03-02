// 2x2 multi-tile GEMM kernel: C[32x32] = A[32xK] @ B[32xK]^T
// Single wave computes 4 output tiles (2x2 grid of 16x16 MFMA tiles).
//
// A/B reuse pattern:
//   A0 (rows 0-15)  reused for C[0,0] and C[0,1]
//   A1 (rows 16-31) reused for C[1,0] and C[1,1]
//   B0 (rows 0-15)  reused for C[0,0] and C[1,0]
//   B1 (rows 16-31) reused for C[0,1] and C[1,1]
//
// Per K iteration: 4 loads (2 A + 2 B), 4 MFMAs
// Total loads = (M_T + N_T) * K_tiles = 4 * K_tiles
// Total MFMAs = M_T * N_T * K_tiles = 4 * K_tiles
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 16)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_f16 = !amdgcn.vgpr<[? + 2]>
!rt_B_f16 = !amdgcn.vgpr<[? + 2]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_multitile_direct target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @get_A_f16(!future_global_read) -> !rt_A_f16
  func.func private @get_B_f16(!future_global_read) -> !rt_B_f16
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // 2x2 multi-tile GEMM kernel (64 threads = 1 wave)
  // Input:  A [32xK f16, row-major], B [32xK f16, row-major]
  // Output: C [32x32 f32, row-major]
  amdgcn.kernel @gemm_multitile_direct arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 128 : index             // 32 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Initialize 4 accumulators to zero (2x2 output tile grid)
    %C00_init = func.call @zero_C() : () -> !rt_C_f32
    %C01_init = func.call @zero_C() : () -> !rt_C_f32
    %C10_init = func.call @zero_C() : () -> !rt_C_f32
    %C11_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: 4 iter_args for 2x2 output tiles
    %C00_final, %C01_final, %C10_final, %C11_final = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(%c00 = %C00_init, %c01 = %C01_init, %c10 = %C10_init, %c11 = %C11_init)
        -> (!rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Load 2 A tiles (M dimension: rows 0-15 and 16-31)
      %A0_fut = func.call @load_A_f16(%A_ptr, %c0, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read
      %A1_fut = func.call @load_A_f16(%A_ptr, %c16, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      // Load 2 B tiles (N dimension: rows 0-15 and 16-31)
      %B0_fut = func.call @load_B_f16(%B_ptr, %c0, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read
      %B1_fut = func.call @load_B_f16(%B_ptr, %c16, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      // Wait for all loads
      %A0 = func.call @get_A_f16(%A0_fut) : (!future_global_read) -> !rt_A_f16
      %A1 = func.call @get_A_f16(%A1_fut) : (!future_global_read) -> !rt_A_f16
      %B0 = func.call @get_B_f16(%B0_fut) : (!future_global_read) -> !rt_B_f16
      %B1 = func.call @get_B_f16(%B1_fut) : (!future_global_read) -> !rt_B_f16

      // 4 MFMAs: C[i][j] += A[i] @ B[j]^T
      %c00_new = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %c00)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c01_new = func.call @mfma_f32_16x16x16_f16(%A0, %B1, %c01)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c10_new = func.call @mfma_f32_16x16x16_f16(%A1, %B0, %c10)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c11_new = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %c11)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %c00_new, %c01_new, %c10_new, %c11_new
          : !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32
    }

    // Store 4 output tiles at their positions in C[32x32]
    //   C[0,0] at (0, 0),  C[0,1] at (0, 16)
    //   C[1,0] at (16, 0), C[1,1] at (16, 16)
    %tok00 = func.call @store_C_f32(%C00_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok01 = func.call @store_C_f32(%C01_final, %C_ptr, %c0, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok10 = func.call @store_C_f32(%C10_final, %C_ptr, %c16, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok11 = func.call @store_C_f32(%C11_final, %C_ptr, %c16, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %tok00 : !write_token
    amdgcn.wait deps %tok01 : !write_token
    amdgcn.wait deps %tok10 : !write_token
    amdgcn.wait deps %tok11 : !write_token

    amdgcn.end_kernel
  }
}
