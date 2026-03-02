// Single-wave 2x2 multi-tile GEMM with LDS + pipelining:
// C[32x32] = A[32xK] @ B[32xK]^T
//
// One wave computes a 2x2 grid of 16x16 MFMA tiles using 4 iter_args.
// Data flows through LDS with XOR swizzle for bank-conflict-free access.
//
// LDS layout per pipeline stage: 4 tiles (2 A rows + 2 B cols), XOR swizzle
//   A0: rows 0-15 of A    -- reused for C[0,0] and C[0,1]
//   A1: rows 16-31 of A   -- reused for C[1,0] and C[1,1]
//   B0: rows 0-15 of B    -- reused for C[0,0] and C[1,0]
//   B1: rows 16-31 of B   -- reused for C[0,1] and C[1,1]
//   Total: 4 x 512 = 2,048 bytes per pipeline stage
//
// Per K iteration: 4 global->LDS loads, 4 LDS->register loads, 4 MFMAs
//
// 2-stage pipeline:
//   stage 0: alloc + load_global_to_lds (async prefetch)
//   stage 1: wait + get_lds + mfma + dealloc
//
// 3-stage pipeline:
//   stage 0: alloc + load_global_to_lds (async prefetch)
//   stage 1: wait + get_lds (consume)
//   stage 2: mfma + dealloc (compute)
//
// Template parameters:
//   {{K}}             - K dimension (must be divisible by 16)
//   {{K_TILES}}       - Number of K tiles = K / 16
//   {{STRIDE_AB}}     - Row stride in bytes for A and B = K * 2
//   {{STAGE_LOAD}}    - Pipeline stage for LDS loads (always 0)
//   {{STAGE_SYNC}}    - Pipeline stage for LDS->reg
//   {{STAGE_COMPUTE}} - Pipeline stage for MFMA

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

amdgcn.module @kittens_gemm_multitile_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir - XOR swizzle mode (async by default)
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // Single-wave 2x2 multi-tile GEMM kernel with pipelined LDS (64 threads = 1 wave)
  // Input:  A [32xK f16, row-major], B [32xK f16, row-major]
  // Output: C [32x32 f32, row-major]
  //
  // LDS: 4 XOR-swizzle tiles x 512 bytes = 2,048 bytes per pipeline stage.
  // The pipeline pass multiplies this by the number of in-flight stages.
  amdgcn.kernel @gemm_multitile_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 2048 : i32} {
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

      // === Stage LOAD: Allocate LDS + cooperative Global -> LDS ===
      // 4 tiles: 2 A (rows 0-15, 16-31) + 2 B (rows 0-15, 16-31)
      %lds_a0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_a1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_A0 = amdgcn.get_lds_offset %lds_a0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_A1 = amdgcn.get_lds_offset %lds_a1_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B0 = amdgcn.get_lds_offset %lds_b0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B1 = amdgcn.get_lds_offset %lds_b1_h {sched.stage = {{STAGE_LOAD}} : i32} : index

      // Load 2 A tiles (M dimension: rows 0-15 and 16-31) into LDS
      %tok_A0 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A0, %A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_A1 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A1, %A_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token

      // Load 2 B tiles (N dimension: rows 0-15 and 16-31) into LDS
      %tok_B0 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B0, %B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B1 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B1, %B_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token

      // === Stage SYNC: Wait for LDS writes + load LDS -> Register ===
      amdgcn.wait deps %tok_A0 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A1 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B0 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B1 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      // No barrier needed for single wave

      %A0_future = func.call @load_lds_A_xor_swizzle_f16(%lds_A0)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      %A1_future = func.call @load_lds_A_xor_swizzle_f16(%lds_A1)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      %B0_future = func.call @load_lds_B_xor_swizzle_f16(%lds_B0)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      %B1_future = func.call @load_lds_B_xor_swizzle_f16(%lds_B1)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read

      // === Stage COMPUTE: 4 MFMAs (2x2 tile grid) ===
      %A0 = func.call @get_lds_A_f16(%A0_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %A1 = func.call @get_lds_A_f16(%A1_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %B0 = func.call @get_lds_B_f16(%B0_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      %B1 = func.call @get_lds_B_f16(%B1_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16

      // C[i][j] += A[i] @ B[j]^T
      %c00_new = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %c00)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c01_new = func.call @mfma_f32_16x16x16_f16(%A0, %B1, %c01)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c10_new = func.call @mfma_f32_16x16x16_f16(%A1, %B0, %c10)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c11_new = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %c11)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at the last stage so the prep pass knows the buffer span.
      amdgcn.dealloc_lds %lds_a0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b1_h {sched.stage = {{STAGE_COMPUTE}} : i32}

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
