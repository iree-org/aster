// Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T  (M = M_WG*M_TILE, N = N_WG*N_TILE)
//
// Uses v_mfma_f32_32x32x8_f16 (8192 ops/inst, 16 AGPRs, 32x32 output tiles, K=8/step).
//
// Dispatched over a flat grid of M_WG * N_WG workgroups. Each workgroup
// contains NUM_WAVES wavefronts that cooperatively load and compute one
// M_TILE x N_TILE output tile:
//   - Each wave loads M_T_PER_WAVE 32-row A tiles (its private slice).
//   - Each wave loads N_T_PER_WAVE 32-row B tiles (a disjoint subset of the B tile).
//   - Loading each 32-row tile uses two 16x64_b half-tile operations.
// After all LDS writes complete, a single s_barrier synchronizes the waves so
// that every wave can read all N_T B tiles that were collectively written.
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
//   {{K_TILES}}        - Total K-steps = K / 32 (outer loop bound)
//   {{M_T}}            - M MFMA tiles per workgroup = M_TILE / 32 (32-row units)
//   {{N_T}}            - N MFMA tiles per workgroup = N_TILE / 32
//   {{M_T_LD}}         - LDS 16-row slots per workgroup = M_TILE / 16 (= 2 * M_T)
//   {{N_T_LD}}         - LDS 16-row slots per workgroup = N_TILE / 16
//   {{M_T_PER_WAVE}}   - 32-row A tiles loaded per wave = M_T / NUM_WAVES
//   {{N_T_PER_WAVE}}   - 32-row B tiles loaded per wave = N_T / NUM_WAVES
//   {{NUM_WAVES}}      - Number of wavefronts (block_dim = NUM_WAVES * 64)
//   {{NUM_M_WAVES}}    - Wave groups in M direction
//   {{NUM_N_WAVES}}    - Wave groups in N direction
//   {{M_T_PW_C}}       - 32-row M tiles per wave for compute = M_T / NUM_M_WAVES
//   {{N_T_PW_C}}       - 32-row N tiles per wave for compute = N_T / NUM_N_WAVES
//   {{M_WG}}           - Workgroup grid size in M = M_total / M_TILE
//   {{N_WG}}           - Workgroup grid size in N = N_total / N_TILE
//   {{SWIZZLE}}        - CTA swizzle group size (must divide M_WG; 1 = no swizzle)
//   {{SWIZZLE_NWG}}    - SWIZZLE * N_WG
//   {{STRIDE_AB}}      - Row stride for A and B in bytes = K * 2
//   {{STRIDE_C}}       - Row stride for C in bytes = N_WG * N_TILE * 4
//   {{A_LDS_BYTES}}    - LDS for all A tiles = K_T * M_T_LD * 1024
//   {{B_LDS_BYTES}}    - LDS for all B tiles = K_T * N_T_LD * 1024
//   K_LOOP_HELPERS_32x32     - Inlined K-loop helper function definitions
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
  // Loading is 1D: wave wid loads M_T_PER_WAVE 32-row A tiles and N_T_PER_WAVE 32-row B tiles.
  // Each 32-row tile is loaded as two 16x64_b half-tiles (upper rows 0-15 and lower rows 16-31).
  // Compute is 2D: wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES.
  // Wave (wm, wn) computes the M_T_PW_C x N_T_PW_C sub-tile (in 32-row units).
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
    %c_K_TILES = arith.constant {{K_TILES}} : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_M_T_LD = arith.constant {{M_T_LD}} : index
    %c_N_T_LD = arith.constant {{N_T_LD}} : index
    %c_M_T_PER_WAVE = arith.constant {{M_T_PER_WAVE}} : index
    %c_N_T_PER_WAVE = arith.constant {{N_T_PER_WAVE}} : index
    %c_M_T_PW_C = arith.constant {{M_T_PW_C}} : index
    %c_N_T_PW_C = arith.constant {{N_T_PW_C}} : index
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

    // Wave decomposition: wid = wm * NUM_N_WAVES + wn (row-major).
    // wave_a_base_16: starting 16-row LDS/global slot for this wave's A load.
    //   = wid * M_T_PER_WAVE * 2  (M_T_PER_WAVE is in 32-row units; *2 for 16-row slots).
    // wave_b_base (32-row units): = wn * N_T_PW_C + wm * N_T_PER_WAVE
    // wave_b_base_16: = wave_b_base * 2 (for LDS addressing).
    %wid = func.call @wave_id() : () -> index
    %wm = affine.apply affine_map<(w) -> (w floordiv {{NUM_N_WAVES}})>(%wid)
    %wn = affine.apply affine_map<(w) -> (w mod {{NUM_N_WAVES}})>(%wid)
    %wave_a_base_16 = affine.apply affine_map<(w)[m] -> (w * m * 2)>(%wid)[%c_M_T_PER_WAVE]
    %wave_b_base = affine.apply
        affine_map<(wn, wm)[npw, nper] -> (wn * npw + wm * nper)>
        (%wn, %wm)[%c_N_T_PW_C, %c_N_T_PER_WAVE]
    %wave_b_base_16 = affine.apply affine_map<(wb) -> (wb * 2)>(%wave_b_base)

    // Compute bases (in 32-row units for compute; *2 for LDS 16-row addressing).
    %wave_a_compute_base = affine.apply affine_map<(w)[mc] -> (w * mc)>(%wm)[%c_M_T_PW_C]
    %wave_b_compute_base = affine.apply affine_map<(w)[nc] -> (w * nc)>(%wn)[%c_N_T_PW_C]
    %wave_a_compute_base_16 = affine.apply affine_map<(w)[mc] -> (w * mc * 2)>(%wm)[%c_M_T_PW_C]
    %wave_b_compute_base_16 = affine.apply affine_map<(w)[nc] -> (w * nc * 2)>(%wn)[%c_N_T_PW_C]

    // Global tile bases for loads: block offset (in 16-row units) + intra-block wave offset.
    // m_global_load_base/n_global_load_base are in 16-row tile units (for @load_global_tile_16x64_b).
    %m_global_load_base = affine.apply
        affine_map<(mwg, wa16)[mtld] -> (mwg * mtld + wa16)>(%m_wg, %wave_a_base_16)[%c_M_T_LD]
    %n_global_load_base = affine.apply
        affine_map<(nwg, wb16)[ntld] -> (nwg * ntld + wb16)>(%n_wg, %wave_b_base_16)[%c_N_T_LD]

    // Global tile bases for C store: block offset + intra-block 2D compute offset.
    // m_global_compute_base/n_global_compute_base are in 32-row/col tile units.
    %m_global_compute_base = affine.apply
        affine_map<(mwg, wmc)[mt] -> (mwg * mt + wmc)>(%m_wg, %wave_a_compute_base)[%c_M_T]
    %n_global_compute_base = affine.apply
        affine_map<(nwg, wnc)[nt] -> (nwg * nt + wnc)>(%n_wg, %wave_b_compute_base)[%c_N_T]

    // K_T * 4 MFMA K-steps per outer iteration (4 per 32-element K-chunk: 32/8 = 4).
    %k_mfma = affine.apply affine_map<()[k] -> (k * 4)>()[%c_K_T]

    // Allocate and zero accumulator buffer: M_T_PW_C x N_T_PW_C output tiles per wave.
    %mn_per_wave = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T_PW_C, %c_N_T_PW_C]
    %C_buf = memref.alloca(%mn_per_wave) : !c_buf_32
    scf.for %i = %c0 to %mn_per_wave step %c1 {
      %z = func.call @zero_C_32x32() : () -> !rt_C_f32_32
      memref.store %z, %C_buf[%i] : !c_buf_32
    } {aster.constexpr}

    // K outer loop: each step processes K_T * 32 K-elements via K_T LDS tile-pairs.
    scf.for %k = %c0 to %c_K_TILES step %c_K_T {
      // === Allocate LDS for this iteration. ===
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // === Issue global loads: each wave loads M_T_PER_WAVE 32-row A tiles and
      //     N_T_PER_WAVE 32-row B tiles. Each 32-row tile = 2 x 16x64_b half-tiles.
      %gfut_a = func.call @k_load_a_32x32_from_global(
          %c_M_T_PER_WAVE, %c_K_T, %A_ptr, %k, %stride_AB, %m_global_load_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_32x32_from_global(
          %c_N_T_PER_WAVE, %c_K_T, %B_ptr, %k, %stride_AB, %n_global_load_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_b_buf

      // === Store global data to LDS (upper + lower half-tiles per 32-row tile). ===
      // wave_a_base_16: 16-row LDS slot index for this wave's A region.
      // c_M_T_LD: total 16-row slots per WG per K-chunk (used to stride across K).
      %tok_a = func.call @k_store_a_32x32_to_lds(
          %c_M_T_PER_WAVE, %c_K_T, %base_a, %wave_a_base_16, %c_M_T_LD, %gfut_a)
          : (index, index, index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_32x32_to_lds(
          %c_N_T_PER_WAVE, %c_K_T, %base_b, %wave_b_base_16, %c_N_T_LD, %gfut_b)
          : (index, index, index, index, index, !gfut_b_buf) -> !tok_b_buf

      // === Wait for this wave's LDS writes to complete, then barrier. ===
      func.call @k_wait_lds_writes_32x32(%tok_a) : (!tok_a_buf) -> ()
      func.call @k_wait_lds_writes_32x32(%tok_b) : (!tok_b_buf) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32, sched.rotate_head}

      // === Read A and B tiles from LDS using 2D compute bases. ===
      // wave_a_compute_base_16: 16-row slot offset for this wave's compute A region.
      %a_fut = func.call @k_read_lds_a_32x32(
          %c_M_T_PW_C, %c_K_T, %base_a, %wave_a_compute_base_16, %c_M_T_LD)
          : (index, index, index, index, index) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_b_32x32(
          %c_N_T_PW_C, %c_K_T, %base_b, %wave_b_compute_base_16, %c_N_T_LD)
          : (index, index, index, index, index) -> !fut_b_buf

      // === Wait for LDS reads and compute M_T_PW_C x N_T_PW_C x k_mfma MFMAs. ===
      func.call @k_wait_and_compute_mfmas_32x32(
          %c_M_T_PW_C, %c_N_T_PW_C, %k_mfma, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf_32) -> ()

      // Release this iteration's LDS buffers.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
    }

    // Store this wave's M_T_PW_C x N_T_PW_C output tiles (in 32-row/col units) to global C.
    func.call @store_c_tiles_32x32(
        %c_M_T_PW_C, %c_N_T_PW_C, %C_buf, %C_ptr, %stride_C, %m_global_compute_base, %n_global_compute_base)
        : (index, index, !c_buf_32, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
