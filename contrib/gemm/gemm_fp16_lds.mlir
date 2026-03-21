// Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T  (M = M_WG*M_TILE, N = N_WG*N_TILE)
//
// Dispatched over a flat grid of M_WG * N_WG workgroups. Each workgroup
// contains NUM_WAVES wavefronts that cooperatively load and compute one
// M_TILE x N_TILE output tile using a hybrid work distribution:
//   - Loading (flat): wave wid loads A_PER_WAVE tiles from the flat (K_T * M_T) A-tile
//     space and B_PER_WAVE tiles from the flat (K_T * N_T) B-tile space.
//   - A single s_barrier synchronizes all waves.
//   - Compute (2D): waves are arranged as a NUM_M_WAVES x NUM_N_WAVES grid.
//     Wave wid maps to (wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES).
//     Wave (wm, wn) reads A rows [wm*M_T_PER_WAVE, (wm+1)*M_T_PER_WAVE) from LDS,
//     B cols [wn*N_T_PER_WAVE, (wn+1)*N_T_PER_WAVE) from LDS, and computes the
//     M_T_PER_WAVE x N_T_PER_WAVE output tiles for its rectangle.
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
//   {{K}}              - K dimension (must be divisible by K_TILE = K_T * 32)
//   {{K_T}}            - K tiles per outer loop step = K_TILE / 32
//   {{K_INNER}}        - MFMA K-steps per K_TILE = K_TILE / 16 (= K_T * 2 for 16x16)
//   {{K_TILES}}        - Total 16x32 K tiles = K / 32 (outer loop bound)
//   {{M_T}}            - M MFMA tiles per workgroup = M_TILE / 16
//   {{N_T}}            - N MFMA tiles per workgroup = N_TILE / 16
//   {{NUM_M_WAVES}}    - Wave grid size in M (num_n_waves = NUM_WAVES / NUM_M_WAVES)
//   {{NUM_N_WAVES}}    - Wave grid size in N = NUM_WAVES / NUM_M_WAVES
//   {{M_T_PER_WAVE}}   - M MFMA tiles per wave = M_T / NUM_M_WAVES
//   {{N_T_PER_WAVE}}   - N MFMA tiles per wave = N_T / NUM_N_WAVES
//   {{TILES_PER_WAVE}} - Output MMA tiles per wave = M_T_PER_WAVE * N_T_PER_WAVE
//   {{A_PER_WAVE}}     - A 16-row tiles loaded per wave = M_T * K_T / NUM_WAVES
//   {{B_PER_WAVE}}     - B 16-row tiles loaded per wave = N_T * K_T / NUM_WAVES
//   {{NUM_WAVES}}      - Number of wavefronts (block_dim = NUM_WAVES * 64)
//   {{M_WG}}           - Workgroup grid size in M = M_total / M_TILE
//   {{N_WG}}           - Workgroup grid size in N = N_total / N_TILE
//   {{SWIZZLE}}        - CTA swizzle group size (must divide M_WG; 1 = no swizzle)
//   {{SWIZZLE_NWG}}    - SWIZZLE * N_WG (embedded in affine maps for modulo/div)
//   {{STRIDE_AB}}      - Row stride for A and B in bytes = K * 2
//   {{STRIDE_C}}       - Row stride for C in bytes = N_WG * N_TILE * 4
//   {{A_LDS_BYTES}}    - LDS for all A tiles = K_T * M_T * 1024
//   {{B_LDS_BYTES}}    - LDS for all B tiles = K_T * N_T * 1024
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
  // Loading: wave wid loads flat A-tile range [wid*A_PER_WAVE, (wid+1)*A_PER_WAVE) and
  //          flat B-tile range [wid*B_PER_WAVE, (wid+1)*B_PER_WAVE).
  // Compute: wave wid -> (wm = wid/NUM_N_WAVES, wn = wid%NUM_N_WAVES) owns the
  //          M_T_PER_WAVE x N_T_PER_WAVE rectangle of output tiles.
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
    %c_K_INNER = arith.constant {{K_INNER}} : index
    %c_K_TILES = arith.constant {{K_TILES}} : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_NUM_N_WAVES = arith.constant {{NUM_N_WAVES}} : index
    %c_M_T_PER_WAVE = arith.constant {{M_T_PER_WAVE}} : index
    %c_N_T_PER_WAVE = arith.constant {{N_T_PER_WAVE}} : index
    %c_TILES_PER_WAVE = arith.constant {{TILES_PER_WAVE}} : index
    %c_A_PER_WAVE = arith.constant {{A_PER_WAVE}} : index
    %c_B_PER_WAVE = arith.constant {{B_PER_WAVE}} : index
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

    // Wave decomposition: flat loading, 2D compute.
    // Loading: wid owns flat A range [wid*A_PER_WAVE, (wid+1)*A_PER_WAVE).
    // Compute: wm = wid / NUM_N_WAVES, wn = wid % NUM_N_WAVES.
    %wid = func.call @wave_id() : () -> index
    %flat_a_start = affine.apply affine_map<(w)[apw] -> (w * apw)>(%wid)[%c_A_PER_WAVE]
    %flat_b_start = affine.apply affine_map<(w)[bpw] -> (w * bpw)>(%wid)[%c_B_PER_WAVE]
    %wm = arith.divui %wid, %c_NUM_N_WAVES : index
    %wn = arith.remui %wid, %c_NUM_N_WAVES : index

    // Global base for loads: block offset in 16-row units (m_t == m_t_ld for 16x16).
    %m_global_base = affine.apply affine_map<(mwg)[mt] -> (mwg * mt)>(%m_wg)[%c_M_T]
    %n_global_base = affine.apply affine_map<(nwg)[nt] -> (nwg * nt)>(%n_wg)[%c_N_T]

    // Allocate and zero accumulator buffer: TILES_PER_WAVE output tiles per wave.
    %C_buf = memref.alloca(%c_TILES_PER_WAVE) : !c_buf
    scf.for %i = %c0 to %c_TILES_PER_WAVE step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // K outer loop: each step processes K_T * 32 K-elements via K_T LDS tiles.
    // LDS is allocated inside the loop so the pipeline can double-buffer (depth 2).
    scf.for %k = %c0 to %c_K_TILES step %c_K_T {
      // === Allocate LDS for this iteration (pipelined: stage GLOBAL_LOAD). ===
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // === Issue global loads: each wave loads its flat slice of A and B tiles. ===
      %gfut_a = func.call @k_load_a_flat(
          %c_A_PER_WAVE, %c_K_T, %c_M_T, %A_ptr, %k, %stride_AB, %m_global_base, %flat_a_start)
          : (index, index, index, !aster_utils.any, index, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_flat(
          %c_B_PER_WAVE, %c_K_T, %c_N_T, %B_ptr, %k, %stride_AB, %n_global_base, %flat_b_start)
          : (index, index, index, !aster_utils.any, index, index, index, index) -> !gfut_b_buf

      // === Store global data to LDS (flat layout, same LDS addresses as before). ===
      %tok_a = func.call @k_store_a_flat(
          %c_A_PER_WAVE, %c_K_T, %c_M_T, %base_a, %flat_a_start, %gfut_a)
          : (index, index, index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_flat(
          %c_B_PER_WAVE, %c_K_T, %c_N_T, %base_b, %flat_b_start, %gfut_b)
          : (index, index, index, index, index, !gfut_b_buf) -> !tok_b_buf

      // === Wait for LDS writes, then barrier. ===
      func.call @k_wait_lds_writes(%tok_a) : (!tok_a_buf) -> ()
      func.call @k_wait_lds_writes(%tok_b) : (!tok_b_buf) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32, sched.rotate_head}

      // === Read tiles from LDS: wave (wm, wn) reads its M/N rectangle. ===
      %a_fut = func.call @k_read_lds_a_2d(
          %c_M_T_PER_WAVE, %c_K_T, %c_M_T, %base_a, %wm)
          : (index, index, index, index, index) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_b_2d(
          %c_N_T_PER_WAVE, %c_K_T, %c_N_T, %base_b, %wn)
          : (index, index, index, index, index) -> !fut_b_buf

      // === Wait for LDS reads and compute M_T_PER_WAVE x N_T_PER_WAVE x K_INNER MFMAs. ===
      func.call @k_wait_and_compute_mfmas_2d(
          %c_M_T_PER_WAVE, %c_N_T_PER_WAVE, %c_K_INNER, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf) -> ()

      // Release this iteration's LDS buffers, then sync all waves.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32, sched.rotate_head}
    }

    // Store this wave's M_T_PER_WAVE x N_T_PER_WAVE output tiles to global C.
    func.call @store_c_tiles_2d(
        %c_M_T_PER_WAVE, %c_N_T_PER_WAVE, %C_buf, %C_ptr, %stride_C,
        %m_wg, %n_wg, %c_M_T, %c_N_T, %wm, %wn)
        : (index, index, !c_buf, !aster_utils.any, index, index, index, index, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
