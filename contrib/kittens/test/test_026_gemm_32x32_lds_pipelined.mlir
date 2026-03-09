// Kittens GEMM kernel with pipelined LDS (32x32x8 MFMA, 32x32 transfer tiles):
// C = A @ B^T
// A: 32xK (f16), B: 32xK (f16), C: 32x32 (f32)
//
// Uses v_mfma_f32_32x32x8_f16: K=8 per MFMA, 32x32 output tile.
// Memory transfer: 32x32 f16 tiles (K=32 per outer iteration, 4 MFMAs each).
// LDS: 2048 bytes per matrix (4 contiguous 32x8 sub-tiles).
//
// 4-stage pipeline via sched.stage annotations:
//   STAGE_GLOBAL_LOAD: alloc LDS, issue 4+4 global loads
//   STAGE_DS_WRITE:    store all global data to LDS
//   STAGE_DS_READ:     wait for LDS writes, read MFMA fragments
//   STAGE_COMPUTE:     4 MFMAs, dealloc LDS
//
// Template parameters:
//   {{K_TILES}}          - Number of K tiles = K / 32
//   {{STRIDE_AB}}        - Row stride in bytes for A and B = K * 2
//   {{STAGE_GLOBAL_LOAD}}, {{STAGE_DS_WRITE}}, {{STAGE_DS_READ}}, {{STAGE_COMPUTE}}

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!ax16 = !amdgcn.agpr<[? + 16]>
!rt_C_f32 = !ax16
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Buffer type aliases (matching lds_32x32_f16.mlir signatures)
!gfut_buf = memref<?x!future_global_read>
!lds_wtok_buf = memref<?x!lds_write_token>
!lds_rfut_buf = memref<?x!future_lds_read>
!wtok_buf = memref<?x!write_token>

amdgcn.module @kittens_gemm_32x32xK_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_32x32_f16.mlir
  func.func private @zero_C_32x32() -> !rt_C_f32
  func.func private @store_C_32x32_f32(!rt_C_f32, !sx2, index, index, index) -> !wtok_buf
  func.func private @wait_global_writes_32x32(!wtok_buf)

  // From kittens/lds_32x32_f16.mlir (32x32 composite primitives)
  func.func private @load_global_tile_32x32_f16(!sx2, index, index, index) -> !gfut_buf
  func.func private @store_global_tile_to_lds_32x32_f16(index, !gfut_buf) -> !lds_wtok_buf
  func.func private @wait_lds_writes_32x32(!lds_wtok_buf)
  func.func private @load_lds_A_32x32_f16(index) -> !lds_rfut_buf
  func.func private @load_lds_B_32x32_f16(index) -> !lds_rfut_buf
  func.func private @compute_mfmas_32x32(!lds_rfut_buf, !lds_rfut_buf, !rt_C_f32) -> !rt_C_f32

  // GEMM kernel with 4-stage pipeline scheduling annotations.
  amdgcn.kernel @gemm_32x32xK_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant 128 : index
    %K_tiles = arith.constant {{K_TILES}} : index

    %C_init = func.call @zero_C_32x32() : () -> !rt_C_f32

    // K-loop with pipeline stage annotations.
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_base = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Stage GLOBAL_LOAD: allocate LDS, issue global loads ===
      %lds_a_h = amdgcn.alloc_lds 2048 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_A = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds 2048 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_B = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      %af = func.call @load_global_tile_32x32_f16(%A_ptr, %c0, %k_base, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !gfut_buf
      %bf = func.call @load_global_tile_32x32_f16(%B_ptr, %c0, %k_base, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !gfut_buf

      // === Stage DS_WRITE: store global data to LDS ===
      %at = func.call @store_global_tile_to_lds_32x32_f16(%lds_A, %af)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !gfut_buf) -> !lds_wtok_buf
      %bt = func.call @store_global_tile_to_lds_32x32_f16(%lds_B, %bf)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !gfut_buf) -> !lds_wtok_buf

      // === Stage DS_READ: wait for LDS writes, read MFMA fragments ===
      func.call @wait_lds_writes_32x32(%at)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (!lds_wtok_buf) -> ()
      func.call @wait_lds_writes_32x32(%bt)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (!lds_wtok_buf) -> ()

      %a_futs = func.call @load_lds_A_32x32_f16(%lds_A)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index) -> !lds_rfut_buf
      %b_futs = func.call @load_lds_B_32x32_f16(%lds_B)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index) -> !lds_rfut_buf

      // === Stage COMPUTE: 4 MFMAs, dealloc LDS ===
      %result = func.call @compute_mfmas_32x32(%a_futs, %b_futs, %acc)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!lds_rfut_buf, !lds_rfut_buf, !rt_C_f32) -> !rt_C_f32

      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %result : !rt_C_f32
    }

    %store_toks = func.call @store_C_32x32_f32(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !wtok_buf
    func.call @wait_global_writes_32x32(%store_toks) : (!wtok_buf) -> ()

    amdgcn.end_kernel
  }
}
