// 4-wave GEMM kernel with pipelined LDS: C[32x32] = A[32xK] @ B[32xK]^T
//
// 2x2 wave grid (waves_m=2, waves_n=2):
//   Wave 0: C[0:16,  0:16]  = A[0:16,  :K] @ B[0:16,  :K]^T
//   Wave 1: C[0:16,  16:32] = A[0:16,  :K] @ B[16:32, :K]^T
//   Wave 2: C[16:32, 0:16]  = A[16:32, :K] @ B[0:16,  :K]^T
//   Wave 3: C[16:32, 16:32] = A[16:32, :K] @ B[16:32, :K]^T
//
// Single-source 1-buffer kernel annotated with sched.stage attributes.
// When aster-scf-pipeline is applied, this produces multi-buffered code.
//
// LDS layout per pipeline stage: 4 tiles (2 A rows + 2 B cols), XOR swizzle
//   A_row0: shared by waves 0,1 (m_offset=0)
//   A_row1: shared by waves 2,3 (m_offset=16)
//   B_col0: shared by waves 0,2 (n_offset=0)
//   B_col1: shared by waves 1,3 (n_offset=16)
//   Each wave redundantly writes its A/B tile (same data -> same LDS address).
//   s_barrier ensures all LDS writes are visible before reads.
//
// 2-stage pipeline:
//   stage 0: alloc + load_global_to_lds (async prefetch)
//   stage 1: wait + barrier + get_lds + mfma + dealloc
//
// 3-stage pipeline:
//   stage 0: alloc + load_global_to_lds (async prefetch)
//   stage 1: wait + barrier + get_lds (consume)
//   stage 2: mfma + dealloc (compute)
//
// Template parameters:
//   {{K}}             - K dimension (must be divisible by 16)
//   {{K_TILES}}       - Number of K tiles = K / 16
//   {{STRIDE_AB}}     - Row stride in bytes for A and B = K * 2
//   {{STAGE_LOAD}}    - Pipeline stage for LDS loads (always 0)
//   {{STAGE_SYNC}}    - Pipeline stage for wait + barrier + LDS->reg
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

amdgcn.module @kittens_gemm_4wave_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir
  func.func private @wave_id() -> index

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

  // 4-wave GEMM kernel (256 threads = 4 waves) with pipelined LDS
  // Input:  A [32xK f16, row-major], B [32xK f16, row-major]
  // Output: C [32x32 f32, row-major]
  //
  // LDS: 4 XOR-swizzle tiles x 512 bytes = 2,048 bytes per pipeline stage.
  // The pipeline pass multiplies this by the number of in-flight stages.
  amdgcn.kernel @gemm_4wave_lds_pipelined arguments <[
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

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 128 : index             // 32 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position in 2x2 grid:
    //   (wave_m, wave_n) = delinearize(wave_id, [2, 2])
    //   m_offset = wave_m * 16   (A row offset, C row offset)
    //   n_offset = wave_n * 16   (B row offset, C col offset)
    %wid = func.call @wave_id() : () -> index
    %c2 = arith.constant 2 : index
    %wave_m, %wave_n = affine.delinearize_index %wid into (%c2, %c2) : index, index
    %m_offset = affine.apply affine_map<()[wm] -> (wm * 16)>()[%wave_m]
    %n_offset = affine.apply affine_map<()[wn] -> (wn * 16)>()[%wave_n]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop with pipeline stage annotations.
    // 4 LDS tiles per stage: A_row0, A_row1 (shared by wave pairs along n),
    // B_col0, B_col1 (shared by wave pairs along m).
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Stage LOAD: Allocate LDS + cooperative Global -> LDS ===
      // 4 tiles: 2 A (one per m-row) + 2 B (one per n-col)
      %lds_a0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_a1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b0_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b1_h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_A0 = amdgcn.get_lds_offset %lds_a0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_A1 = amdgcn.get_lds_offset %lds_a1_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B0 = amdgcn.get_lds_offset %lds_b0_h {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B1 = amdgcn.get_lds_offset %lds_b1_h {sched.stage = {{STAGE_LOAD}} : i32} : index

      // Select this wave's A buffer: wave_m=0 -> A0, wave_m=1 -> A1
      %lds_A_stride = affine.apply affine_map<()[a0, a1] -> (a1 - a0)>()[%lds_A0, %lds_A1]
      %lds_A = affine.apply affine_map<()[base, wm, stride] -> (base + wm * stride)>
          ()[%lds_A0, %wave_m, %lds_A_stride]

      // Select this wave's B buffer: wave_n=0 -> B0, wave_n=1 -> B1
      %lds_B_stride = affine.apply affine_map<()[b0, b1] -> (b1 - b0)>()[%lds_B0, %lds_B1]
      %lds_B = affine.apply affine_map<()[base, wn, stride] -> (base + wn * stride)>
          ()[%lds_B0, %wave_n, %lds_B_stride]

      // Each wave cooperatively loads its A and B tiles.
      // Waves sharing the same m-row redundantly write the same A data;
      // waves sharing the same n-col redundantly write the same B data.
      %tok_A = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A, %A_ptr, %m_offset, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      %tok_B = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B, %B_ptr, %n_offset, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token

      // === Stage SYNC: Wait for LDS writes + barrier + load LDS -> Register ===
      amdgcn.wait deps %tok_A {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      // Cross-wave barrier: ensures all 4 waves' LDS writes are visible
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_SYNC}} : i32}

      %A_future = func.call @load_lds_A_xor_swizzle_f16(%lds_A)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      %B_future = func.call @load_lds_B_xor_swizzle_f16(%lds_B)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read

      // === Stage COMPUTE: MFMA ===
      %A_tile = func.call @get_lds_A_f16(%A_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %B_tile = func.call @get_lds_B_f16(%B_future)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at the last stage so the prep pass knows the buffer span.
      amdgcn.dealloc_lds %lds_a0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b1_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's (m, n) offset in C
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %m_offset, %n_offset, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
