// Kittens GEMM kernel with pipelining annotations: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Single-source 1-buffer kernel annotated with sched.stage attributes.
// When aster-scf-pipeline is applied, this should produce code equivalent
// to the hand-written 2buf and 3buf versions.
//
// 2-stage pipeline (equivalent to 2buf):
//   stage 0: load_global_to_lds (async prefetch)
//   stage 1: lds_to_reg + mfma (consume + compute)
//
// 3-stage pipeline (equivalent to 3buf):
//   stage 0: load_global_to_lds (async prefetch)
//   stage 1: lds_to_reg (consume)
//   stage 2: mfma (compute)
//
// Template parameters:
//   {{K}}             - K dimension (e.g., 32, 64, 128)
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

amdgcn.module @kittens_gemm_16x16xK_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_transfers.mlir - XOR swizzle mode
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_xor_swizzle_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_xor_swizzle_f16(index) -> !rt_B_f16

  // GEMM kernel with pipeline scheduling annotations.
  //
  // The loop body is a simple 1-buffer pattern, but sched.stage attributes
  // tell the aster-scf-pipeline pass how to transform it:
  //
  //   STAGE_LOAD=0, STAGE_SYNC=1, STAGE_COMPUTE=1 -> 2-stage (double buffer)
  //   STAGE_LOAD=0, STAGE_SYNC=1, STAGE_COMPUTE=2 -> 3-stage (triple buffer)
  //
  // The pass will generate prologue/kernel/epilogue sections, allocating
  // separate LDS buffers per in-flight iteration automatically.
  amdgcn.kernel @gemm_16x16xK_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 1024 : i32} {
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

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop with pipeline stage annotations.
    //
    // Raw alloc_lds/dealloc_lds ops with sched.stage let the prep pass
    // hoist N copies before the loop and create rotating offset iter_args.
    // N = STAGE_COMPUTE - STAGE_LOAD + 1 (2 for double-buffer, 3 for triple).
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Allocate LDS inside the loop: one buffer each for A and B tiles.
      // The prep pass will hoist N copies of each and rotate offsets.
      // XOR swizzle: 512 bytes per tile (16x16 f16, no padding)
      %lds_a_handle = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_b_handle = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      %lds_A = amdgcn.get_lds_offset %lds_a_handle {sched.stage = {{STAGE_LOAD}} : i32} : index
      %lds_B = amdgcn.get_lds_offset %lds_b_handle {sched.stage = {{STAGE_LOAD}} : i32} : index

      // === Stage 0: Cooperative load Global -> LDS ===
      func.call @load_global_to_lds_xor_swizzle_f16(%lds_A, %A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> ()
      func.call @load_global_to_lds_xor_swizzle_f16(%lds_B, %B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> ()

      // === Stage SYNC: LDS -> Register ===
      %A_tile = func.call @load_lds_to_register_A_xor_swizzle_f16(%lds_A)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !rt_A_f16
      %B_tile = func.call @load_lds_to_register_B_xor_swizzle_f16(%lds_B)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !rt_B_f16

      // === Stage COMPUTE: MFMA ===
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at the last stage so the prep pass knows the buffer span.
      amdgcn.dealloc_lds %lds_a_handle {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_handle {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result to global memory
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
