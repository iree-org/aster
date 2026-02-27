// 2-wave GEMM kernel: C[32x16] = A[32xK] @ B[16xK]^T
//
// 2x1 wave grid (waves_m=2, waves_n=1):
//   Wave 0: C[0:16,  0:16] = A[0:16,  :K] @ B[0:16, :K]^T
//   Wave 1: C[16:32, 0:16] = A[16:32, :K] @ B[0:16, :K]^T
//
// Both waves share the same B matrix. Each wave offsets its A load and C store
// by wave_id * 16 rows. Uses direct global loads (no LDS).
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 16)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_2wave target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir
  func.func private @wave_id() -> index

  // From kittens/global_16x16_f16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16_future(!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // 2-wave GEMM kernel (128 threads = 2 waves)
  // Input:  A [32xK f16, row-major], B [16xK f16, row-major]
  // Output: C [32x16 f32, row-major]
  amdgcn.kernel @gemm_2wave arguments <[
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
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position in 2x1 grid: m_offset = wave_id * 16
    %wid = func.call @wave_id() : () -> index
    %m_offset = affine.apply affine_map<()[wid] -> (wid * 16)>()[%wid]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: each wave iterates over K tiles independently
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Load A tile at this wave's row offset
      %A_fut = func.call @load_A_f16(%A_ptr, %m_offset, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      // Load B tile (shared across waves, always from row 0)
      %B_fut = func.call @load_B_f16(%B_ptr, %c0, %k_offset, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read

      // Wait for loads and compute: acc += A_tile @ B_tile^T
      %new_acc = func.call @mfma_f32_16x16x16_f16_future(%A_fut, %B_fut, %acc)
          : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's row offset in C
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %m_offset, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
