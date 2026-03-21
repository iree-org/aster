// Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T  (M = M_WG*M_TILE, N = N_WG*N_TILE)
//
// Uses v_mfma_f32_32x32x8_f16 (8192 ops/inst, 16 AGPRs, 32x32 output tiles, K=8/step).
//
// Dispatched over a flat grid of M_WG * N_WG workgroups. Each workgroup
// contains NUM_WAVES wavefronts that cooperatively load and compute one
// M_TILE x N_TILE output tile using a hybrid work distribution:
//   - Loading (flat): wave wid loads A_PER_WAVE 16-row half-tiles from the flat
//     (K_T * M_T_LD) space and B_PER_WAVE half-tiles from (K_T * N_T_LD).
//     M_T_LD = M_TILE / 16 = 2 * M_T (two 16-row slots per 32-row MFMA tile).
//   - A single s_barrier synchronizes all waves.
//   - Compute (2D): waves are arranged as a NUM_M_WAVES x NUM_N_WAVES grid.
//     Wave wid maps to (wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES).
//     Wave (wm, wn) reads A rows [wm*M_T_PER_WAVE, (wm+1)*M_T_PER_WAVE) from LDS,
//     B cols [wn*N_T_PER_WAVE, (wn+1)*N_T_PER_WAVE) from LDS, and computes the
//     M_T_PER_WAVE x N_T_PER_WAVE output tiles for its rectangle.
//
// CTA swizzle: same formula as gemm_fp16_lds.mlir.
//
// Tile geometry:
//   MFMA tile:     32x32 (v_mfma_f32_32x32x8_f16, K=8 per step)
//   Transfer tile: 16x32 f16 elements per half-tile (dwordx4 global load)
//   LDS tile:      2 x 1024 bytes per 32-row MFMA tile
//
// Template parameters:
//   {{K}}              - K dimension (must be divisible by K_TILE = K_T * 32)
//   {{K_T}}            - K tiles per outer loop step = K_TILE / 32
//   {{K_INNER}}        - MFMA K-steps per K_TILE = K_TILE / 8 (= K_T * 4 for 32x32)
//   {{K_TILES}}        - Total K-steps = K / 32 (outer loop bound)
//   {{M_T}}            - M MFMA tiles per workgroup = M_TILE / 32 (32-row units)
//   {{N_T}}            - N MFMA tiles per workgroup = N_TILE / 32
//   {{M_T_LD}}         - LDS 16-row slots per workgroup = M_TILE / 16 (= 2 * M_T)
//   {{N_T_LD}}         - LDS 16-row slots per workgroup = N_TILE / 16
//   {{NUM_M_WAVES}}    - Wave grid size in M (num_n_waves = NUM_WAVES / NUM_M_WAVES)
//   {{NUM_N_WAVES}}    - Wave grid size in N = NUM_WAVES / NUM_M_WAVES
//   {{M_T_PER_WAVE}}   - M MFMA tiles per wave = M_T / NUM_M_WAVES (32-row units)
//   {{N_T_PER_WAVE}}   - N MFMA tiles per wave = N_T / NUM_N_WAVES
//   {{TILES_PER_WAVE}} - Output MMA tiles per wave = M_T_PER_WAVE * N_T_PER_WAVE
//   {{A_PER_WAVE}}     - A 16-row half-tiles loaded per wave = M_T_LD * K_T / NUM_WAVES
//   {{B_PER_WAVE}}     - B 16-row half-tiles loaded per wave = N_T_LD * K_T / NUM_WAVES
//   {{NUM_WAVES}}      - Number of wavefronts (block_dim = NUM_WAVES * 64)
//   {{M_WG}}           - Workgroup grid size in M = M_total / M_TILE
//   {{N_WG}}           - Workgroup grid size in N = N_total / N_TILE
//   {{SWIZZLE}}        - CTA swizzle group size (must divide M_WG; 1 = no swizzle)
//   {{SWIZZLE_NWG}}    - SWIZZLE * N_WG
//   {{STRIDE_AB}}      - Row stride for A and B in bytes = K * 2
//   {{STRIDE_C}}       - Row stride for C in bytes = N_WG * N_TILE * 4
//   {{A_LDS_BYTES}}    - LDS for all A tiles = K_T * M_T_LD * 1024
//   {{B_LDS_BYTES}}    - LDS for all B tiles = K_T * N_T_LD * 1024
//   K_LOOP_HELPERS_32x32 - Inlined K-loop helper function definitions
//
// Stage annotations (set all to 0 for non-pipelined execution):
//   {{STAGE_GLOBAL_LOAD}}, {{STAGE_DS_WRITE}}, {{STAGE_DS_READ}}, {{STAGE_COMPUTE}}

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!ax16 = !amdgcn.agpr<[? + 16]>
!rt_A_f16_32 = !vx2
!rt_B_f16_32 = !vx2
!rt_C_f32_32 = !ax16
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Memref buffer types used by K-loop helpers (dynamic sizes, constexpr-expanded).
!gfut_a_buf = memref<?x!future_global_read>
!gfut_b_buf = memref<?x!future_global_read>
!tok_a_buf = memref<?x!lds_write_token>
!tok_b_buf = memref<?x!lds_write_token>
!fut_a_buf = memref<?x!future_lds_read>
!fut_b_buf = memref<?x!future_lds_read>
!c_buf_32 = memref<?x!rt_C_f32_32>

amdgcn.module @gemm_fp16_lds_32x32 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir.
  func.func private @wave_id() -> index
  // From global_16x64_b.mlir / indexing_ptr.mlir.
  func.func private @prepare_ptr(!sx2) -> !aster_utils.any
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read
  // From lds_16x64_b.mlir.
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  // From mlir_kernels/library/common/futures.mlir.
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From compute_32x32_f16.mlir (AGPR accumulators, 16 AGPRs).
  func.func private @zero_C_32x32() -> !rt_C_f32_32
  func.func private @mfma_f32_32x32x8_f16(!rt_A_f16_32, !rt_B_f16_32, !rt_C_f32_32) -> !rt_C_f32_32
  func.func private @store_global_C_mfma_f32_32x32x8_f16(!rt_C_f32_32, !aster_utils.any, index, index, index)

  // Inlined K-loop helper functions (from gemm_32x32_f16_k_loop_helpers.mlir).
  {{K_LOOP_HELPERS_32x32}}

  // Multi-CU GEMM kernel using v_mfma_f32_32x32x8_f16.
  // Block (m_wg, n_wg) computes C[m_wg*M_TILE:(m_wg+1)*M_TILE, n_wg*N_TILE:(n_wg+1)*N_TILE].
  // Loading: wave wid loads flat A half-tile range [wid*A_PER_WAVE, (wid+1)*A_PER_WAVE) and
  //          flat B half-tile range [wid*B_PER_WAVE, (wid+1)*B_PER_WAVE).
  // Compute: wave wid -> (wm = wid/NUM_N_WAVES, wn = wid%NUM_N_WAVES) owns the
  //          M_T_PER_WAVE x N_T_PER_WAVE rectangle of output tiles.
  amdgcn.kernel @gemm_fp16_lds_32x32 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %A_raw = amdgcn.load_arg 0 : !sx2
    %B_raw = amdgcn.load_arg 1 : !sx2
    %C_raw = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %A_ptr = func.call @prepare_ptr(%A_raw) : (!sx2) -> !aster_utils.any
    %B_ptr = func.call @prepare_ptr(%B_raw) : (!sx2) -> !aster_utils.any
    %C_ptr = func.call @prepare_ptr(%C_raw) : (!sx2) -> !aster_utils.any

    // Constants.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_K_T = arith.constant {{K_T}} : index
    %c_K_INNER = arith.constant {{K_INNER}} : index
    %c_K_TILES = arith.constant {{K_TILES}} : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_M_T_LD = arith.constant {{M_T_LD}} : index
    %c_N_T_LD = arith.constant {{N_T_LD}} : index
    %c_NUM_N_WAVES = arith.constant {{NUM_N_WAVES}} : index
    %c_M_T_PER_WAVE = arith.constant {{M_T_PER_WAVE}} : index
    %c_N_T_PER_WAVE = arith.constant {{N_T_PER_WAVE}} : index
    %c_TILES_PER_WAVE = arith.constant {{TILES_PER_WAVE}} : index
    %c_A_PER_WAVE = arith.constant {{A_PER_WAVE}} : index
    %c_B_PER_WAVE = arith.constant {{B_PER_WAVE}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index

    // CTA swizzle: same formula as gemm_fp16_lds.mlir.
    %flat_id = gpu.block_id x
    %m_wg = affine.apply
        affine_map<(fid) -> ((fid floordiv {{SWIZZLE_NWG}}) * {{SWIZZLE}} + fid mod {{SWIZZLE}})>
        (%flat_id)
    %n_wg = affine.apply
        affine_map<(fid) -> ((fid mod {{SWIZZLE_NWG}}) floordiv {{SWIZZLE}})>
        (%flat_id)

    // Wave decomposition: flat loading, 2D compute.
    // Loading: wid owns flat A range [wid*A_PER_WAVE, (wid+1)*A_PER_WAVE).
    // Compute: wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES.
    %wid = func.call @wave_id() : () -> index
    %flat_a_start = affine.apply affine_map<(w)[apw] -> (w * apw)>(%wid)[%c_A_PER_WAVE]
    %flat_b_start = affine.apply affine_map<(w)[bpw] -> (w * bpw)>(%wid)[%c_B_PER_WAVE]
    %wm = arith.divui %wid, %c_NUM_N_WAVES : index
    %wn = arith.remui %wid, %c_NUM_N_WAVES : index

    // Global base for loads: block offset in 16-row units (= m_wg * m_t_ld).
    %m_global_base = affine.apply affine_map<(mwg)[mtld] -> (mwg * mtld)>(%m_wg)[%c_M_T_LD]
    %n_global_base = affine.apply affine_map<(nwg)[ntld] -> (nwg * ntld)>(%n_wg)[%c_N_T_LD]

    // Allocate and zero accumulator buffer: TILES_PER_WAVE output tiles per wave.
    %C_buf = memref.alloca(%c_TILES_PER_WAVE) : !c_buf_32
    scf.for %i = %c0 to %c_TILES_PER_WAVE step %c1 {
      %z = func.call @zero_C_32x32() : () -> !rt_C_f32_32
      memref.store %z, %C_buf[%i] : !c_buf_32
    } {aster.constexpr}

    // K outer loop.
    scf.for %k = %c0 to %c_K_TILES step %c_K_T {
      // === Allocate LDS for this iteration. ===
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // === Issue global loads: flat 16-row half-tile ranges. ===
      %gfut_a = func.call @k_load_a_flat_32x32(
          %c_A_PER_WAVE, %c_K_T, %c_M_T_LD, %A_ptr, %k, %stride_AB, %m_global_base, %flat_a_start)
          : (index, index, index, !aster_utils.any, index, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_flat_32x32(
          %c_B_PER_WAVE, %c_K_T, %c_N_T_LD, %B_ptr, %k, %stride_AB, %n_global_base, %flat_b_start)
          : (index, index, index, !aster_utils.any, index, index, index, index) -> !gfut_b_buf

      // === Store global data to LDS. ===
      %tok_a = func.call @k_store_a_flat_32x32(
          %c_A_PER_WAVE, %c_K_T, %c_M_T_LD, %base_a, %flat_a_start, %gfut_a)
          : (index, index, index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_flat_32x32(
          %c_B_PER_WAVE, %c_K_T, %c_N_T_LD, %base_b, %flat_b_start, %gfut_b)
          : (index, index, index, index, index, !gfut_b_buf) -> !tok_b_buf

      // === Wait for LDS writes, then barrier. ===
      func.call @k_wait_lds_writes_32x32(%tok_a) : (!tok_a_buf) -> ()
      func.call @k_wait_lds_writes_32x32(%tok_b) : (!tok_b_buf) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32, sched.rotate_head}

      // === Read tiles from LDS: wave (wm, wn) reads its M/N rectangle. ===
      %a_fut = func.call @k_read_lds_a_2d_32x32(
          %c_M_T_PER_WAVE, %c_K_T, %c_M_T_LD, %base_a, %wm)
          : (index, index, index, index, index) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_b_2d_32x32(
          %c_N_T_PER_WAVE, %c_K_T, %c_N_T_LD, %base_b, %wn)
          : (index, index, index, index, index) -> !fut_b_buf

      // === Wait for LDS reads and compute M_T_PER_WAVE x N_T_PER_WAVE x K_INNER MFMAs. ===
      func.call @k_wait_and_compute_mfmas_2d_32x32(
          %c_M_T_PER_WAVE, %c_N_T_PER_WAVE, %c_K_INNER, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf_32) -> ()

      // Release this iteration's LDS buffers.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32, sched.rotate_head}
    }

    // Store this wave's M_T_PER_WAVE x N_T_PER_WAVE output tiles to global C.
    func.call @store_c_tiles_2d_32x32(
        %c_M_T_PER_WAVE, %c_N_T_PER_WAVE, %C_buf, %C_ptr, %stride_C,
        %m_wg, %n_wg, %c_M_T, %c_N_T, %wm, %wn)
        : (index, index, !c_buf_32, !aster_utils.any, index, index, index, index, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
