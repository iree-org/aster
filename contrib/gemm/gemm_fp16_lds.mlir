// Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T  (M = M_WG*M_TILE, N = N_WG*N_TILE)
//
// Dispatched over a flat grid of M_WG * N_WG workgroups. Each workgroup
// contains NUM_WAVES wavefronts that cooperatively load and compute one
// M_TILE x N_TILE output tile:
//   - Each wave loads M_T_PER_WAVE rows of A (its private slice).
//   - Each wave loads N_T_PER_WAVE rows of B (a disjoint subset of the B tile).
// After all LDS writes complete, a single s_barrier synchronizes the waves so
// that every wave can read all N_T B tiles that were collectively written.
//
// CTA swizzle: SWIZZLE consecutive flat block IDs are assigned to the same
// N-block but different M-blocks, so that those blocks all reuse the same B
// tile from L2 cache before moving to the next N-block.
//   n_wg = (flat_id mod SWIZZLE_NWG) floordiv SWIZZLE
//   m_wg = (flat_id floordiv SWIZZLE_NWG) * SWIZZLE + flat_id mod SWIZZLE
// (SWIZZLE_NWG = SWIZZLE * N_WG).  With SWIZZLE=1 this reduces to row-major.
//
// Tile geometry:
//   MFMA tile:     16x16 (v_mfma_f32_16x16x16_f16)
//   Transfer tile: 16x32 f16 elements (dwordx4 global load)
//   LDS tile:      1024 bytes per 16x32 transfer tile
//
// Template parameters:
//   {{K}}            - K dimension (must be divisible by K_TILE = K_T * 32)
//   {{K_T}}          - K tiles per outer loop step = K_TILE / 32
//   {{K_TILES}}      - Total 16x32 K tiles = K / 32 (outer loop bound)
//   {{M_T}}          - M MFMA tiles per workgroup = M_TILE / 16
//   {{N_T}}          - N MFMA tiles per workgroup = N_TILE / 16
//   {{M_T_PER_WAVE}} - M tiles loaded per wave (1D) = M_T / NUM_WAVES
//   {{N_T_PER_WAVE}} - B tiles loaded per wave (1D) = N_T / NUM_WAVES
//   {{NUM_WAVES}}    - Number of wavefronts (block_dim = NUM_WAVES * 64)
//   {{NUM_M_WAVES}}  - Wave groups in M direction (NUM_M_WAVES * NUM_N_WAVES = NUM_WAVES)
//   {{NUM_N_WAVES}}  - Wave groups in N direction
//   {{M_T_PW_C}}     - M tiles per wave for compute (2D) = M_T / NUM_M_WAVES
//   {{N_T_PW_C}}     - N tiles per wave for compute (2D) = N_T / NUM_N_WAVES
//   {{M_WG}}         - Workgroup grid size in M = M_total / M_TILE
//   {{N_WG}}         - Workgroup grid size in N = N_total / N_TILE
//   {{SWIZZLE}}      - CTA swizzle group size (must divide M_WG; 1 = no swizzle)
//   {{SWIZZLE_NWG}}  - SWIZZLE * N_WG (embedded in affine maps for modulo/div)
//   {{STRIDE_AB}}    - Row stride for A and B in bytes = K * 2
//   {{STRIDE_C}}     - Row stride for C in bytes = N_WG * N_TILE * 4
//   {{A_LDS_BYTES}}  - LDS for all A tiles = K_T * M_T * 1024
//   {{B_LDS_BYTES}}  - LDS for all B tiles = K_T * N_T * 1024
//   K_LOOP_HELPERS     - Inlined K-loop helper function definitions
//
// Stage annotations (set all to 0 for non-pipelined execution):
//   {{STAGE_GLOBAL_LOAD}}, {{STAGE_DS_WRITE}}, {{STAGE_DS_READ}}, {{STAGE_COMPUTE}}

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!ax4 = !amdgcn.agpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4
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
!vals_a_buf = memref<?x!rt_A_f16>
!vals_b_buf = memref<?x!rt_B_f16>
!c_buf = memref<?x!rt_C_f32>

amdgcn.module @gemm_fp16_lds target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir.
  func.func private @wave_id() -> index
  // From global_16x64_b.mlir / indexing_ptr.mlir.
  func.func private @prepare_ptr(!sx2) -> !aster_utils.any
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read
  // From lds_16x64_b.mlir.
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  // From lds_mfma_16x64_b.mlir.
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From compute_16x16_f16.mlir (AGPR accumulators).
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !aster_utils.any, index, index, index)

  // Inlined K-loop helper functions (from gemm_16x32_f16_k_loop_helpers.mlir).
  {{K_LOOP_HELPERS}}

  // Multi-CU GEMM kernel.
  // Block (m_wg, n_wg) computes C[m_wg*M_TILE:(m_wg+1)*M_TILE, n_wg*N_TILE:(n_wg+1)*N_TILE].
  // Loading is 1D: wave wid loads M_T_PER_WAVE A rows and N_T_PER_WAVE B cols.
  // Compute is 2D: wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES.
  // Wave (wm, wn) computes the M_T_PW_C x N_T_PW_C sub-tile starting at
  // (wm*M_T_PW_C, wn*N_T_PW_C) within the block's output tile.
  amdgcn.kernel @gemm_fp16_lds arguments <[
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
    %c_M_T_PER_WAVE = arith.constant {{M_T_PER_WAVE}} : index
    %c_N_T_PER_WAVE = arith.constant {{N_T_PER_WAVE}} : index
    %c_M_T_PW_C = arith.constant {{M_T_PW_C}} : index
    %c_N_T_PW_C = arith.constant {{N_T_PW_C}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index

    // CTA swizzle: delinearize the flat block ID into (m_wg, n_wg) so that
    // SWIZZLE consecutive block IDs map to the same N-block, maximizing B L2 reuse.
    //   n_wg = (flat_id mod SWIZZLE_NWG) floordiv SWIZZLE
    //   m_wg = (flat_id floordiv SWIZZLE_NWG) * SWIZZLE + flat_id mod SWIZZLE
    // SWIZZLE=1 reduces to standard row-major (m_wg = flat_id / N_WG, n_wg = flat_id % N_WG).
    %flat_id = gpu.block_id x
    %m_wg = affine.apply
        affine_map<(fid) -> ((fid floordiv {{SWIZZLE_NWG}}) * {{SWIZZLE}} + fid mod {{SWIZZLE}})>
        (%flat_id)
    %n_wg = affine.apply
        affine_map<(fid) -> ((fid mod {{SWIZZLE_NWG}}) floordiv {{SWIZZLE}})>
        (%flat_id)

    // Wave decomposition within the block: wid = wm * NUM_N_WAVES + wn (row-major).
    // Loading: each wm-group cooperatively loads A[wm*M_T_PW_C .. (wm+1)*M_T_PW_C);
    //   wn selects the M_T_PER_WAVE-sized slice within that range, giving
    //   wave_a_base = wm*M_T_PW_C + wn*M_T_PER_WAVE = wid*M_T_PER_WAVE.
    // Each wn-group cooperatively loads B[wn*N_T_PW_C .. (wn+1)*N_T_PW_C);
    //   wm selects the N_T_PER_WAVE-sized slice within that range, giving
    //   wave_b_base = wn*N_T_PW_C + wm*N_T_PER_WAVE (NOT wid*N_T_PER_WAVE).
    // Compute (2D): wave (wm, wn) reads A rows [wm*M_T_PW_C, ...) and B cols
    //   [wn*N_T_PW_C, ...) from LDS and computes the M_T_PW_C x N_T_PW_C sub-tile of C.
    %wid = func.call @wave_id() : () -> index
    %wm = affine.apply affine_map<(w) -> (w floordiv {{NUM_N_WAVES}})>(%wid)
    %wn = affine.apply affine_map<(w) -> (w mod {{NUM_N_WAVES}})>(%wid)
    %wave_a_base = affine.apply affine_map<(w)[m] -> (w * m)>(%wid)[%c_M_T_PER_WAVE]
    %wave_b_base = affine.apply
        affine_map<(wn, wm)[npw, nper] -> (wn * npw + wm * nper)>
        (%wn, %wm)[%c_N_T_PW_C, %c_N_T_PER_WAVE]
    %wave_a_compute_base = affine.apply affine_map<(w)[mc] -> (w * mc)>(%wm)[%c_M_T_PW_C]
    %wave_b_compute_base = affine.apply affine_map<(w)[nc] -> (w * nc)>(%wn)[%c_N_T_PW_C]

    // Global tile bases for loads: block offset + intra-block 1D wave offset.
    // m_global_load_base: starting M tile (in 16-row units) for this wave's A load.
    // n_global_load_base: starting N tile for this wave's B load in global B.
    %m_global_load_base = affine.apply
        affine_map<(mwg, wa)[mt] -> (mwg * mt + wa)>(%m_wg, %wave_a_base)[%c_M_T]
    %n_global_load_base = affine.apply
        affine_map<(nwg, wb)[nt] -> (nwg * nt + wb)>(%n_wg, %wave_b_base)[%c_N_T]

    // Global tile bases for C store: block offset + intra-block 2D compute offset.
    // m_global_compute_base: starting M tile for wave (wm, wn)'s C rows.
    // n_global_compute_base: starting N tile for wave (wm, wn)'s C cols.
    %m_global_compute_base = affine.apply
        affine_map<(mwg, wmc)[mt] -> (mwg * mt + wmc)>(%m_wg, %wave_a_compute_base)[%c_M_T]
    %n_global_compute_base = affine.apply
        affine_map<(nwg, wnc)[nt] -> (nwg * nt + wnc)>(%n_wg, %wave_b_compute_base)[%c_N_T]

    // Allocate shared LDS: A tiles for all waves, B tiles for all waves.
    // Each wave writes its own non-overlapping A region and a disjoint slice of B.
    // After a barrier, all waves read all B tiles written cooperatively.
    %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}}
    %base_a = amdgcn.get_lds_offset %lds_a_h : index
    %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}}
    %base_b = amdgcn.get_lds_offset %lds_b_h : index

    // K_T*2 MFMA K-steps per outer iteration (2 per 16x32 transfer tile).
    %k_mfma = affine.apply affine_map<()[k] -> (k * 2)>()[%c_K_T]

    // Allocate and zero accumulator buffer: M_T_PW_C x N_T_PW_C output tiles per wave (2D partition).
    %mn_per_wave = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T_PW_C, %c_N_T_PW_C]
    %C_buf = memref.alloca(%mn_per_wave) : !c_buf
    scf.for %i = %c0 to %mn_per_wave step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // K outer loop: each step processes K_T * 32 K-elements via K_T LDS tiles.
    scf.for %k = %c0 to %c_K_TILES step %c_K_T {
      // === Issue global loads cooperatively: each wave loads its A rows and B slice. ===
      %gfut_a = func.call @k_load_a_16x32_from_global(
          %c_M_T_PER_WAVE, %c_K_T, %A_ptr, %k, %stride_AB, %m_global_load_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_16x32_from_global(
          %c_N_T_PER_WAVE, %c_K_T, %B_ptr, %k, %stride_AB, %n_global_load_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_b_buf

      // === Store global data to LDS. ===
      // A: wave i writes to tiles [wave_a_base, wave_a_base + M_T_PER_WAVE) per K slice.
      // B: wave i writes to tiles [wave_b_base, wave_b_base + N_T_PER_WAVE) per K slice;
      //    tiles_per_slice = N_T preserves the shared LDS layout for cross-wave reads.
      %tok_a = func.call @k_store_a_16x32_to_lds(
          %c_M_T_PER_WAVE, %c_K_T, %base_a, %wave_a_base, %c_M_T, %gfut_a)
          : (index, index, index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_16x32_to_lds(
          %c_N_T_PER_WAVE, %c_K_T, %base_b, %wave_b_base, %c_N_T, %gfut_b)
          : (index, index, index, index, index, !gfut_b_buf) -> !tok_b_buf

      // === Wait for this wave's LDS writes to complete, then barrier. ===
      // The s_barrier ensures all waves have finished their B slice before any
      // wave reads B tiles written by other waves.
      func.call @k_wait_lds_writes(%tok_a) : (!tok_a_buf) -> ()
      func.call @k_wait_lds_writes(%tok_b) : (!tok_b_buf) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier>

      // === Read A and B tiles from LDS using 2D compute bases. ===
      // Each wave reads only its M_T_PW_C A rows (at wave_a_compute_base) and
      // N_T_PW_C B cols (at wave_b_compute_base); tiles_per_slice stays at M_T/N_T
      // to index into the shared LDS layout written cooperatively by all waves.
      %a_fut = func.call @k_read_lds_a(
          %c_M_T_PW_C, %c_K_T, %base_a, %wave_a_compute_base, %c_M_T)
          : (index, index, index, index, index) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_b(
          %c_N_T_PW_C, %c_K_T, %base_b, %wave_b_compute_base, %c_N_T)
          : (index, index, index, index, index) -> !fut_b_buf

      // === Wait for LDS reads and compute M_T_PW_C x N_T_PW_C x k_mfma MFMAs. ===
      func.call @k_wait_and_compute_mfmas(
          %c_M_T_PW_C, %c_N_T_PW_C, %k_mfma, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier>
    }

    // Store this wave's M_T_PW_C x N_T_PW_C output tiles to global C.
    // m_base = block offset + wm*M_T_PW_C, n_base = block offset + wn*N_T_PW_C.
    func.call @store_c_tiles(
        %c_M_T_PW_C, %c_N_T_PW_C, %C_buf, %C_ptr, %stride_C, %m_global_compute_base, %n_global_compute_base)
        : (index, index, !c_buf, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
