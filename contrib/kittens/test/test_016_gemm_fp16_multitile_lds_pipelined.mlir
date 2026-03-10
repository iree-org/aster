// Single-wave 2x2 multi-tile GEMM with LDS (16x32 tiles) + pipelining + AGPR:
// C[32x32] = A[32xK] @ B[32xK]^T
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
//
// LDS layout per pipeline stage: 4 tiles (2 A rows + 2 B cols), 1024 bytes each
//   A0: rows 0-15 of A, A1: rows 16-31 of A
//   B0: rows 0-15 of B, B1: rows 16-31 of B
//   Total: 4 x 1024 = 4,096 bytes per pipeline stage
//
// Per K iteration: 4 global loads, 4 LDS stores (8 tokens), 8 LDS reads, 8 MFMAs
//   (2 MFMAs per tile x 4 tile combinations)
//
// 4-stage pipeline:
//   STAGE_GLOBAL_LOAD: alloc + dwordx4 global loads
//   STAGE_DS_WRITE:    store to LDS (2 tokens per tile)
//   STAGE_DS_READ:     wait + read K0/K1 sub-tiles
//   STAGE_COMPUTE:     8 MFMAs (4 tile combos x 2 K-steps) + dealloc
//
// Template parameters:
//   {{K}}, {{K_TILES}}, {{STRIDE_AB}}
//   {{STAGE_GLOBAL_LOAD}}, {{STAGE_DS_WRITE}}, {{STAGE_DS_READ}}, {{STAGE_COMPUTE}}

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

amdgcn.module @kittens_gemm_multitile_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
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

  amdgcn.kernel @gemm_multitile_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index  // bytes per f16 element
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index  // K1 byte offset within LDS row
    %c16 = arith.constant 16 : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant 128 : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // Initialize 4 accumulators (2x2 output tile grid)
    %C00_init = func.call @zero_C() : () -> !rt_C_f32
    %C01_init = func.call @zero_C() : () -> !rt_C_f32
    %C10_init = func.call @zero_C() : () -> !rt_C_f32
    %C11_init = func.call @zero_C() : () -> !rt_C_f32

    %C00_final, %C01_final, %C10_final, %C11_final = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(%c00 = %C00_init, %c01 = %C01_init, %c10 = %C10_init, %c11 = %C11_init)
        -> (!rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Stage GLOBAL_LOAD: Allocate LDS + issue global loads ===
      %lds_a0_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_a1_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_b0_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_b1_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}

      %A0_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      %A1_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      %B0_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      %B1_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c16, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read

      // === Stage DS_WRITE: Store to LDS (2 tokens per tile = 8 total) ===
      %lds_A0 = amdgcn.get_lds_offset %lds_a0_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_A1 = amdgcn.get_lds_offset %lds_a1_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_B0 = amdgcn.get_lds_offset %lds_b0_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_B1 = amdgcn.get_lds_offset %lds_b1_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index

      %tA0_0, %tA0_1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A0, %A0_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tA1_0, %tA1_1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A1, %A1_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tB0_0, %tB0_1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B0, %B0_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tB1_0, %tB1_1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B1, %B1_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)

      // === Stage DS_READ: Wait + read K0/K1 sub-tiles ===
      amdgcn.wait deps %tA0_0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tA0_1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tA1_0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tA1_1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tB0_0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tB0_1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tB1_0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tB1_1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token

      // K0 sub-tiles (byte offset 0 within LDS row)
      %A0_K0_fut = func.call @load_lds_A_swizzled(%lds_A0, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %A1_K0_fut = func.call @load_lds_A_swizzled(%lds_A1, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B0_K0_fut = func.call @load_lds_B_swizzled(%lds_B0, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B1_K0_fut = func.call @load_lds_B_swizzled(%lds_B1, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // K1 sub-tiles (byte offset 32 within LDS row)
      %A0_K1_fut = func.call @load_lds_A_swizzled(%lds_A0, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %A1_K1_fut = func.call @load_lds_A_swizzled(%lds_A1, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B0_K1_fut = func.call @load_lds_B_swizzled(%lds_B0, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B1_K1_fut = func.call @load_lds_B_swizzled(%lds_B1, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // === Stage COMPUTE: 8 MFMAs (4 tile combos x 2 K-steps) ===
      // Extract K0 values
      %A0_K0 = func.call @get_lds_read_value_vx2(%A0_K0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %A1_K0 = func.call @get_lds_read_value_vx2(%A1_K0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B0_K0 = func.call @get_lds_read_value_vx2(%B0_K0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %B1_K0 = func.call @get_lds_read_value_vx2(%B1_K0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16

      // K0 MFMAs: C[i][j] += A[i]_K0 @ B[j]_K0^T
      %c00_k0 = func.call @mfma_f32_16x16x16_f16(%A0_K0, %B0_K0, %c00)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c01_k0 = func.call @mfma_f32_16x16x16_f16(%A0_K0, %B1_K0, %c01)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c10_k0 = func.call @mfma_f32_16x16x16_f16(%A1_K0, %B0_K0, %c10)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c11_k0 = func.call @mfma_f32_16x16x16_f16(%A1_K0, %B1_K0, %c11)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Extract K1 values
      %A0_K1 = func.call @get_lds_read_value_vx2(%A0_K1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %A1_K1 = func.call @get_lds_read_value_vx2(%A1_K1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B0_K1 = func.call @get_lds_read_value_vx2(%B0_K1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %B1_K1 = func.call @get_lds_read_value_vx2(%B1_K1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16

      // K1 MFMAs: C[i][j] += A[i]_K1 @ B[j]_K1^T
      %c00_k1 = func.call @mfma_f32_16x16x16_f16(%A0_K1, %B0_K1, %c00_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c01_k1 = func.call @mfma_f32_16x16x16_f16(%A0_K1, %B1_K1, %c01_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c10_k1 = func.call @mfma_f32_16x16x16_f16(%A1_K1, %B0_K1, %c10_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      %c11_k1 = func.call @mfma_f32_16x16x16_f16(%A1_K1, %B1_K1, %c11_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Dealloc at last stage
      amdgcn.dealloc_lds %lds_a0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b1_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %c00_k1, %c01_k1, %c10_k1, %c11_k1
          : !rt_C_f32, !rt_C_f32, !rt_C_f32, !rt_C_f32
    }

    // Fire-and-forget store 4 output tiles
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C00_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C01_final, %C_ptr, %c0, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C10_final, %C_ptr, %c16, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C11_final, %C_ptr, %c16, %c16, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
