// 2-wave FP8 GEMM kernel: C[32x16] = A[32xK] @ B[16xK]^T
//
// 2x1 wave grid (waves_m=2, waves_n=1):
//   Wave 0: C[0:16,  0:16] = A[0:16,  :K] @ B[0:16, :K]^T
//   Wave 1: C[16:32, 0:16] = A[16:32, :K] @ B[0:16, :K]^T
//
// Both waves share the same B matrix. Each wave offsets its A load and C store
// by wave_id * 16 rows. Uses direct global loads (no LDS).
//
// Key differences from FP16 2-wave:
//   - K step is 32 (vs 16 for f16) because FP8 MFMA processes 32 elements
//   - Element size is 1 byte (vs 2 for f16), so stride_AB = K * 1
//   - Uses v_mfma_f32_16x16x32_fp8_fp8 (vs v_mfma_f32_16x16x16_f16)
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 32)
//   {{K_TILES}}   - Number of K tiles = K / 32
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 1

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_fp8 = !vx2
!rt_B_fp8 = !vx2
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_fp8_2wave target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir
  func.func private @wave_id() -> index

  // From kittens/tiles_16x16_fp8.mlir
  func.func private @load_A_fp8(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_fp8(!sx2, index, index, index) -> !future_global_read
  func.func private @get_A_fp8(!future_global_read) -> !rt_A_fp8
  func.func private @get_B_fp8(!future_global_read) -> !rt_B_fp8
  func.func private @zero_C_fp8() -> !rt_C_f32
  func.func private @mfma_f32_16x16x32_fp8_fp8(!rt_A_fp8, !rt_B_fp8, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_fp8(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // 2-wave FP8 GEMM kernel (128 threads = 2 waves)
  // Input:  A [32xK fp8, row-major], B [16xK fp8, row-major]
  // Output: C [32x16 f32, row-major]
  amdgcn.kernel @gemm_fp8_2wave arguments <[
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

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 1 byte per fp8
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 32)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position in 2x1 grid: m_offset = wave_id * 16
    %wid = func.call @wave_id() : () -> index
    %m_offset = affine.apply affine_map<()[wid] -> (wid * 16)>()[%wid]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C_fp8() : () -> !rt_C_f32

    // K-loop: each wave iterates over K tiles independently
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      // k_offset = k * 32 (column offset in elements)
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // Load A tile at this wave's row offset
      %A_future = func.call @load_A_fp8(%A_ptr, %m_offset, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      // Load B tile (shared across waves, always from row 0)
      %B_future = func.call @load_B_fp8(%B_ptr, %c0, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      %A_tile = func.call @get_A_fp8(%A_future) : (!future_global_read) -> !rt_A_fp8
      %B_tile = func.call @get_B_fp8(%B_future) : (!future_global_read) -> !rt_B_fp8

      // MFMA: acc += A_tile @ B_tile^T
      %new_acc = func.call @mfma_f32_16x16x32_fp8_fp8(%A_tile, %B_tile, %acc)
          : (!rt_A_fp8, !rt_B_fp8, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's row offset in C
    %store_tok = func.call @store_C_fp8(%C_final, %C_ptr, %m_offset, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
