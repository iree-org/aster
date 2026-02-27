// Kittens GEMM kernel with double-buffer LDS: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// This implements latency hiding through double buffering:
// - While computing iteration k, prefetch k+1 data into alternate buffer
// - Overlaps global memory latency (~400 cycles) with compute (~16 cycles)
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

amdgcn.module @kittens_gemm_16x16xK_lds_2buf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir
  func.func private @alloc_lds_2buffer_padded() -> (index, index, index, index)

  // From kittens/lds_16x16_f16.mlir
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_f16(index) -> !future_lds_read
  func.func private @load_lds_B_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using double-buffer LDS
  //
  // Memory flow (steady state):
  //   Cycle 0:    Prefetch k+1: Global -> LDS[pong] (async, 400 cycles)
  //   Cycle 0:    Wait for k's data in LDS[ping]
  //   Cycle 10:   Load k: LDS[ping] -> Register (~10 cycles)
  //   Cycle 20:   Compute k: MFMA (~16 cycles)
  //   Cycle 36:   Next iteration
  //   (Background: k+1 load completes by cycle 400, ready for next iteration)
  //
  // Total: ~36 cycles/iteration (compute bound - 11x speedup!)
  amdgcn.kernel @gemm_16x16xK_lds_2buf arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 2176 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: double buffer for A and B tiles
    // A[0], B[0]: ping buffers (offsets 0, 544)
    // A[1], B[1]: pong buffers (offsets 1088, 1632)
    %lds_A0, %lds_B0, %lds_A1, %lds_B1 = func.call @alloc_lds_2buffer_padded()
        : () -> (index, index, index, index)

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // === Prefetch iteration 0 ===
    // Load first iteration's data into buffer 0 (ping)
    %pf_tok_A0 = func.call @load_global_to_lds_f16(%lds_A0, %A_ptr, %c0, %c0, %stride_AB)
        : (index, !sx2, index, index, index) -> !lds_write_token
    amdgcn.wait deps %pf_tok_A0 : !lds_write_token
    %pf_tok_B0 = func.call @load_global_to_lds_f16(%lds_B0, %B_ptr, %c0, %c0, %stride_AB)
        : (index, !sx2, index, index, index) -> !lds_write_token
    amdgcn.wait deps %pf_tok_B0 : !lds_write_token

    // K-loop: double-buffered iteration over K dimension
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      // === Buffer Selection (Ping-Pong) ===
      // buf_idx = k % 2
      %buf_idx = arith.remui %k, %c2 : index
      %is_ping = arith.cmpi eq, %buf_idx, %c0 : index

      // Select current buffers (where k's data is)
      %lds_A_cur = arith.select %is_ping, %lds_A0, %lds_A1 : index
      %lds_B_cur = arith.select %is_ping, %lds_B0, %lds_B1 : index

      // Select next buffers (where k+1's data will go)
      %lds_A_next = arith.select %is_ping, %lds_A1, %lds_A0 : index
      %lds_B_next = arith.select %is_ping, %lds_B1, %lds_B0 : index

      // === Prefetch k+1 (if not last iteration) ===
      %k_next = arith.addi %k, %c1 : index
      %has_next = arith.cmpi ult, %k_next, %K_tiles : index

      scf.if %has_next {
        // Compute k+1 offset
        %k_next_offset = arith.muli %k_next, %c16 : index

        // Load k+1 to next buffer (wait immediately for sync behavior)
        %pf_tok_A = func.call @load_global_to_lds_f16(%lds_A_next, %A_ptr, %c0, %k_next_offset, %stride_AB)
            : (index, !sx2, index, index, index) -> !lds_write_token
        amdgcn.wait deps %pf_tok_A : !lds_write_token
        %pf_tok_B = func.call @load_global_to_lds_f16(%lds_B_next, %B_ptr, %c0, %k_next_offset, %stride_AB)
            : (index, !sx2, index, index, index) -> !lds_write_token
        amdgcn.wait deps %pf_tok_B : !lds_write_token
      }

      // When we have multi-waves we should also barrier but we'll need to be
      // careful about impacts on pipelining + async.
      //   amdgcn.sopp.sopp <s_barrier>

      // === Load LDS -> Register ===
      // Each thread loads its portion from current LDS buffer
      %A_future = func.call @load_lds_A_f16(%lds_A_cur)
          : (index) -> !future_lds_read
      %A_tile = func.call @get_lds_A_f16(%A_future)
          : (!future_lds_read) -> !rt_A_f16
      %B_future = func.call @load_lds_B_f16(%lds_B_cur)
          : (index) -> !future_lds_read
      %B_tile = func.call @get_lds_B_f16(%B_future)
          : (!future_lds_read) -> !rt_B_f16

      // === Compute ===
      // MFMA: acc += A_tile @ B_tile^T
      // While this computes, k+1's data loads in background (if has_next)
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
