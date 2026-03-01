// Single-wave 4x4 multi-tile GEMM with LDS + pipelining:
// C[64x64] = A[64xK] @ B[64xK]^T
//
// One wave computes a 4x4 grid of 16x16 MFMA tiles using 16 iter_args.
// Data flows through LDS with XOR swizzle for bank-conflict-free access.
//
// LDS layout per pipeline stage: 8 tiles (4 A rows + 4 B cols), XOR swizzle
//   A0-A3: rows 0-15, 16-31, 32-47, 48-63 of A
//   B0-B3: rows 0-15, 16-31, 32-47, 48-63 of B
//   Total: 8 x 512 = 4,096 bytes per pipeline stage
//
// Per K iteration: 8 global->LDS loads, 8 LDS->register loads, 16 MFMAs
// A/B reuse: A[i] reused across 4 N tiles, B[j] reused across 4 M tiles
//   -> 8 loads amortized across 16 MFMAs (2:1 compute:load ratio)
//
// Register budget:
//   16 C accumulators x 4 VGPRs = 64 VGPRs (accumulators)
//   8 A/B tiles x 2 VGPRs = 16 VGPRs (peak operand registers)
//   + temporaries -> ~92 VGPRs estimated
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

amdgcn.module @kittens_gemm_multitile_4x4_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir - XOR swizzle mode
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // Single-wave 4x4 multi-tile GEMM kernel with pipelined LDS (64 threads = 1 wave)
  // Input:  A [64xK f16, row-major], B [64xK f16, row-major]
  // Output: C [64x64 f32, row-major]
  //
  // LDS: 8 XOR-swizzle tiles x 512 bytes = 4,096 bytes per pipeline stage.
  amdgcn.kernel @gemm_multitile_4x4_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 4096 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %row32 = arith.constant 32 : index
    %row48 = arith.constant 48 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 256 : index             // 64 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Initialize 16 accumulators to zero (4x4 output tile grid)
    %C00_init = func.call @zero_C() : () -> !rt_C_f32
    %C01_init = func.call @zero_C() : () -> !rt_C_f32
    %C02_init = func.call @zero_C() : () -> !rt_C_f32
    %C03_init = func.call @zero_C() : () -> !rt_C_f32
    %C10_init = func.call @zero_C() : () -> !rt_C_f32
    %C11_init = func.call @zero_C() : () -> !rt_C_f32
    %C12_init = func.call @zero_C() : () -> !rt_C_f32
    %C13_init = func.call @zero_C() : () -> !rt_C_f32
    %C20_init = func.call @zero_C() : () -> !rt_C_f32
    %C21_init = func.call @zero_C() : () -> !rt_C_f32
    %C22_init = func.call @zero_C() : () -> !rt_C_f32
    %C23_init = func.call @zero_C() : () -> !rt_C_f32
    %C30_init = func.call @zero_C() : () -> !rt_C_f32
    %C31_init = func.call @zero_C() : () -> !rt_C_f32
    %C32_init = func.call @zero_C() : () -> !rt_C_f32
    %C33_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: 16 iter_args for 4x4 output tiles
    %C00_f, %C01_f, %C02_f, %C03_f,
    %C10_f, %C11_f, %C12_f, %C13_f,
    %C20_f, %C21_f, %C22_f, %C23_f,
    %C30_f, %C31_f, %C32_f, %C33_f = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(
            %c00 = %C00_init, %c01 = %C01_init, %c02 = %C02_init, %c03 = %C03_init,
            %c10 = %C10_init, %c11 = %C11_init, %c12 = %C12_init, %c13 = %C13_init,
            %c20 = %C20_init, %c21 = %C21_init, %c22 = %C22_init, %c23 = %C23_init,
            %c30 = %C30_init, %c31 = %C31_init, %c32 = %C32_init, %c33 = %C33_init
        ) -> (!rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
              !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
              !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
              !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Stage LOAD: Allocate 8 LDS tiles + cooperative Global -> LDS ===
      %lds_a0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_a1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_a2_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_a3_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b2_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b3_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_A0 = amdgcn.get_lds_offset %lds_a0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_A1 = amdgcn.get_lds_offset %lds_a1_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_A2 = amdgcn.get_lds_offset %lds_a2_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_A3 = amdgcn.get_lds_offset %lds_a3_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B0 = amdgcn.get_lds_offset %lds_b0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B1 = amdgcn.get_lds_offset %lds_b1_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B2 = amdgcn.get_lds_offset %lds_b2_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B3 = amdgcn.get_lds_offset %lds_b3_h {sched.stage = {{STAGE_LOAD}} : i32} : index

      // Load 4 A tiles (rows 0-15, 16-31, 32-47, 48-63)
      %tok_A0 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A0, %A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_A1 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A1, %A_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_A2 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A2, %A_ptr, %row32, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_A3 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A3, %A_ptr, %row48, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token

      // Load 4 B tiles (rows 0-15, 16-31, 32-47, 48-63)
      %tok_B0 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B0, %B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B1 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B1, %B_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B2 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B2, %B_ptr, %row32, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B3 = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B3, %B_ptr, %row48, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token

      // === Stage SYNC: Wait for LDS writes + load LDS -> Register ===
      amdgcn.wait deps %tok_A0 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A1 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A2 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A3 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B0 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B1 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B2 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B3 {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token

      %A0_fut = func.call @load_lds_A_xor_swizzle_f16(%lds_A0)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %A1_fut = func.call @load_lds_A_xor_swizzle_f16(%lds_A1)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %A2_fut = func.call @load_lds_A_xor_swizzle_f16(%lds_A2)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %A3_fut = func.call @load_lds_A_xor_swizzle_f16(%lds_A3)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %B0_fut = func.call @load_lds_B_xor_swizzle_f16(%lds_B0)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %B1_fut = func.call @load_lds_B_xor_swizzle_f16(%lds_B1)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %B2_fut = func.call @load_lds_B_xor_swizzle_f16(%lds_B2)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read
      %B3_fut = func.call @load_lds_B_xor_swizzle_f16(%lds_B3)
          {sched.stage = {{STAGE_SYNC}} : i32} : (index) -> !future_lds_read

      // === Stage COMPUTE: Extract register tiles ===
      %A0 = func.call @get_lds_A_f16(%A0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %A1 = func.call @get_lds_A_f16(%A1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %A2 = func.call @get_lds_A_f16(%A2_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %A3 = func.call @get_lds_A_f16(%A3_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B0 = func.call @get_lds_B_f16(%B0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %B1 = func.call @get_lds_B_f16(%B1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %B2 = func.call @get_lds_B_f16(%B2_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %B3 = func.call @get_lds_B_f16(%B3_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16

      // 16 MFMAs: C[i][j] += A[i] @ B[j]^T
      // Row 0: A0 x B0..B3
      %c00_new = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %c00)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c01_new = func.call @mfma_f32_16x16x16_f16(%A0, %B1, %c01)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c02_new = func.call @mfma_f32_16x16x16_f16(%A0, %B2, %c02)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c03_new = func.call @mfma_f32_16x16x16_f16(%A0, %B3, %c03)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      // Row 1: A1 x B0..B3
      %c10_new = func.call @mfma_f32_16x16x16_f16(%A1, %B0, %c10)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c11_new = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %c11)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c12_new = func.call @mfma_f32_16x16x16_f16(%A1, %B2, %c12)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c13_new = func.call @mfma_f32_16x16x16_f16(%A1, %B3, %c13)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      // Row 2: A2 x B0..B3
      %c20_new = func.call @mfma_f32_16x16x16_f16(%A2, %B0, %c20)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c21_new = func.call @mfma_f32_16x16x16_f16(%A2, %B1, %c21)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c22_new = func.call @mfma_f32_16x16x16_f16(%A2, %B2, %c22)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c23_new = func.call @mfma_f32_16x16x16_f16(%A2, %B3, %c23)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      // Row 3: A3 x B0..B3
      %c30_new = func.call @mfma_f32_16x16x16_f16(%A3, %B0, %c30)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c31_new = func.call @mfma_f32_16x16x16_f16(%A3, %B1, %c31)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c32_new = func.call @mfma_f32_16x16x16_f16(%A3, %B2, %c32)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c33_new = func.call @mfma_f32_16x16x16_f16(%A3, %B3, %c33)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at the last stage
      amdgcn.dealloc_lds %lds_a0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a2_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a3_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b2_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b3_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %c00_new, %c01_new, %c02_new, %c03_new,
                %c10_new, %c11_new, %c12_new, %c13_new,
                %c20_new, %c21_new, %c22_new, %c23_new,
                %c30_new, %c31_new, %c32_new, %c33_new
          : !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
            !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
            !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32,
            !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32
    }

    // Store 16 output tiles at their positions in C[64x64]
    // Row 0
    %tok00 = func.call @store_C_f32(%C00_f, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok01 = func.call @store_C_f32(%C01_f, %C_ptr, %c0, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok02 = func.call @store_C_f32(%C02_f, %C_ptr, %c0, %row32, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok03 = func.call @store_C_f32(%C03_f, %C_ptr, %c0, %row48, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    // Row 1
    %tok10 = func.call @store_C_f32(%C10_f, %C_ptr, %c16, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok11 = func.call @store_C_f32(%C11_f, %C_ptr, %c16, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok12 = func.call @store_C_f32(%C12_f, %C_ptr, %c16, %row32, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok13 = func.call @store_C_f32(%C13_f, %C_ptr, %c16, %row48, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    // Row 2
    %tok20 = func.call @store_C_f32(%C20_f, %C_ptr, %row32, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok21 = func.call @store_C_f32(%C21_f, %C_ptr, %row32, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok22 = func.call @store_C_f32(%C22_f, %C_ptr, %row32, %row32, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok23 = func.call @store_C_f32(%C23_f, %C_ptr, %row32, %row48, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    // Row 3
    %tok30 = func.call @store_C_f32(%C30_f, %C_ptr, %row48, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok31 = func.call @store_C_f32(%C31_f, %C_ptr, %row48, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok32 = func.call @store_C_f32(%C32_f, %C_ptr, %row48, %row32, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    %tok33 = func.call @store_C_f32(%C33_f, %C_ptr, %row48, %row48, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    // Wait for all stores
    amdgcn.wait deps %tok00 : !write_token
    amdgcn.wait deps %tok01 : !write_token
    amdgcn.wait deps %tok02 : !write_token
    amdgcn.wait deps %tok03 : !write_token
    amdgcn.wait deps %tok10 : !write_token
    amdgcn.wait deps %tok11 : !write_token
    amdgcn.wait deps %tok12 : !write_token
    amdgcn.wait deps %tok13 : !write_token
    amdgcn.wait deps %tok20 : !write_token
    amdgcn.wait deps %tok21 : !write_token
    amdgcn.wait deps %tok22 : !write_token
    amdgcn.wait deps %tok23 : !write_token
    amdgcn.wait deps %tok30 : !write_token
    amdgcn.wait deps %tok31 : !write_token
    amdgcn.wait deps %tok32 : !write_token
    amdgcn.wait deps %tok33 : !write_token

    amdgcn.end_kernel
  }
}
