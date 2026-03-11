// Kittens GEMM kernel with pipelining annotations + AGPR: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
//
// 4-stage pipeline:
//   STAGE_GLOBAL_LOAD: alloc LDS + issue dwordx4 global loads
//   STAGE_DS_WRITE:    wait for global loads, store to LDS (2 tokens per tile)
//   STAGE_DS_READ:     wait for LDS writes, read K0/K1 sub-tiles
//   STAGE_COMPUTE:     2 MFMAs per tile (K0 + K1), dealloc LDS
//
// Template parameters:
//   {{K}}                  - K dimension (must be divisible by 32)
//   {{K_TILES}}            - Number of K tiles = K / 32
//   {{STRIDE_AB}}          - Row stride in bytes for A and B = K * 2
//   {{STAGE_GLOBAL_LOAD}}  - Pipeline stage for global loads (always 0)
//   {{STAGE_DS_WRITE}}     - Pipeline stage for LDS writes
//   {{STAGE_DS_READ}}      - Pipeline stage for LDS reads
//   {{STAGE_COMPUTE}}      - Pipeline stage for MFMA

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.module @kittens_gemm_16x16xK_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From compute_16x16_f16.mlir (AGPR)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !sx2, index, index, index)

  // From lds_16x64_b.mlir
  func.func private @load_global_tile_16x64_b(!sx2, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @load_lds_A_swizzled(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled(index, index, index) -> !future_lds_read
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  // GEMM kernel with 4-stage pipeline scheduling annotations + AGPR accumulators.
  amdgcn.kernel @gemm_16x16xK_lds_pipelined arguments <[
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
    %c2 = arith.constant 2 : index  // bytes per f16 element
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index  // K1 byte offset within LDS row

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 32)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop with 4-stage pipeline annotations
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Stage GLOBAL_LOAD: Allocate LDS + issue dwordx4 global loads ===
      %lds_a_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_b_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}

      %A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      %B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read

      // === Stage DS_WRITE: Store global data to LDS (2 tokens per tile) ===
      %lds_A = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_B = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index

      %tok_A0, %tok_A1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A, %A_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_B0, %tok_B1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B, %B_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)

      // === Stage DS_READ: Wait for LDS writes + read K0/K1 sub-tiles ===
      amdgcn.wait deps %tok_A0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token

      // K0 sub-tile at byte offset 0 within LDS row
      %A0_fut = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B0_fut = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // K1 sub-tile at byte offset 32 within LDS row
      %A1_fut = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B1_fut = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // === Stage COMPUTE: 2 MFMAs (K0 + K1) ===
      %A0 = func.call @get_lds_read_value_vx2(%A0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B0 = func.call @get_lds_read_value_vx2(%B0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %acc_k0 = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %acc)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      %A1 = func.call @get_lds_read_value_vx2(%A1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B1 = func.call @get_lds_read_value_vx2(%B1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %acc_k1 = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %acc_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at last stage
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %acc_k1 : !rt_C_f32
    }

    // Fire-and-forget store
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
