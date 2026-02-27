// Kittens GEMM kernel with single-buffer LDS: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// This is the baseline LDS implementation to establish correctness.
// Uses single buffer (no latency hiding) - memory bound performance.
//
// Template parameters:
//   {{K}}         - K dimension (e.g., 32, 64, 128)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.module @kittens_gemm_16x16xK_lds_1buf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir
  func.func private @alloc_lds_1buffer_padded() -> (index, index)

  // From kittens/lds_16x16_f16.mlir
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_f16(index) -> !future_lds_read
  func.func private @load_lds_B_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using single-buffer LDS
  //
  // Memory flow per iteration:
  //   1. Cooperative load: Global -> LDS (all threads, ~400 cycles)
  //   2. Wait: Wait for LDS writes to complete
  //   3. Load: LDS -> Register (per-thread, ~10 cycles)
  //   4. Compute: MFMA (16 cycles)
  //   Total: ~426 cycles/iteration (memory bound)
  amdgcn.kernel @gemm_16x16xK_lds_1buf arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 1088 : i32} {
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
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: single buffer for A and B tiles
    %lds_A, %lds_B = func.call @alloc_lds_1buffer_padded() : () -> (index, index)

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: iterate over K dimension in 16-element tiles
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      // k_offset = k * 16 (column offset in elements)
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Step 1: Cooperative load Global -> LDS ===
      // All threads cooperatively load A and B tiles from global memory to LDS
      %tok_A = func.call @load_global_to_lds_f16(%lds_A, %A_ptr, %c0, %k_offset, %stride_AB)
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B = func.call @load_global_to_lds_f16(%lds_B, %B_ptr, %c0, %k_offset, %stride_AB)
          : (index, !sx2, index, index, index) -> !lds_write_token

      // === Step 2: Wait ===
      // Wait for this thread's LDS writes
      amdgcn.wait deps %tok_A : !lds_write_token
      amdgcn.wait deps %tok_B : !lds_write_token
      // When we have multi-waves we should also barrier but we'll need to be
      // careful about impacts on pipelining + async.
      //   amdgcn.sopp.sopp <s_barrier>

      // === Step 3: Load LDS -> Register ===
      // Each thread loads its portion from LDS into register tiles
      %A_future = func.call @load_lds_A_f16(%lds_A)
          : (index) -> !future_lds_read
      %A_tile = func.call @get_lds_A_f16(%A_future)
          : (!future_lds_read) -> !rt_A_f16
      %B_future = func.call @load_lds_B_f16(%lds_B)
          : (index) -> !future_lds_read
      %B_tile = func.call @get_lds_B_f16(%B_future)
          : (!future_lds_read) -> !rt_B_f16

      // === Step 4: Compute ===
      // MFMA: acc += A_tile @ B_tile^T
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result to global memory
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
