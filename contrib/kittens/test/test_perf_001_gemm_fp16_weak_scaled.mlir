// Multi-workgroup multi-wave constexpr GEMM with LDS + pipelining
// Uses 16x16x16 MFMA with dwordx4 global loads (16x32 transfer tiles).
// Each 16x32 tile yields 2 MFMA K-steps: 2x global load throughput vs dwordx2.

// Register type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx4 = !amdgcn.sgpr<[? + 4]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!ax4 = !amdgcn.agpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!index_pair = !aster_utils.struct<i: index, j: index>

// Memref buffer type aliases (used in helper function signatures)
// Dynamic sizes -- SROA + constexpr expansion specialize to static.
!gfut_a_buf = memref<?x!future_global_read>
!gfut_b_buf = memref<?x!future_global_read>
!tok_a_buf = memref<?x!lds_write_token>
!tok_b_buf = memref<?x!lds_write_token>
!fut_a_buf = memref<?x!future_lds_read>
!fut_b_buf = memref<?x!future_lds_read>
!vals_a_buf = memref<?x!rt_A_f16>
!vals_b_buf = memref<?x!rt_B_f16>
!c_buf = memref<?x!rt_C_f32>

amdgcn.module @kittens_gemm_f16_weak_scaled target = #amdgcn.target<gfx942> {
  // Library functions (external, provided by preload library)
  func.func private @wave_id() -> index
  // From compute_16x16_f16.mlir (AGPR MFMA and fire-and-forget C tile store)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !aster_utils.any, index, index, index)
  // From indexing.mlir
  func.func private @linear_block_id() -> index
  // From global_16x64_b.mlir / lds_16x64_b.mlir (dwordx4 global load + 16x32 LDS operations)
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From global_16x64_b[_buf].mlir -- type-erases raw !sx2 arg to !aster_utils.any
  func.func private @prepare_ptr(!sx2) -> !aster_utils.any
{{K_LOOP_HELPERS}}

  // Multi-WG multi-wave GEMM with pipelined LDS (16x16x16 MFMA, dwordx4 loads)
  // M_WAVES * N_WAVES waves per WG; block_dim = (M_WAVES * N_WAVES * 64, 1, 1).
  // num_blocks = M_WG * N_WG; flat block ID delinearized into (m_wg, n_wg).
  // wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
  amdgcn.kernel @gemm_f16_weak_scaled arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{SHARED_MEM}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %A_raw = amdgcn.load_arg 0 : !sx2
    %B_raw = amdgcn.load_arg 1 : !sx2
    %C_raw = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %A_rsrc = func.call @prepare_ptr(%A_raw) : (!sx2) -> !aster_utils.any
    %B_rsrc = func.call @prepare_ptr(%B_raw) : (!sx2) -> !aster_utils.any
    %C_rsrc = func.call @prepare_ptr(%C_raw) : (!sx2) -> !aster_utils.any
    // Constants
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index  // bytes per f16 element
    %c1 = arith.constant 1 : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_K_T = arith.constant {{K_T}} : index
    %c_M_TILES_WG = arith.constant {{M_TILES_WG}} : index
    %c_N_TILES_WG = arith.constant {{N_TILES_WG}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // WG tile offsets via layout
    %bid = func.call @linear_block_id() : () -> index
    %wg_m_off = layout.linearize %bid,
        #layout.strided_layout<[{{M_WG}}, {{N_WG}}] : [{{M_TILES_WG}}, 0]>
    %wg_n_off = layout.linearize %bid,
        #layout.strided_layout<[{{M_WG}}, {{N_WG}}] : [0, {{N_TILES_WG}}]>

    // Wave COMPUTE distribution: which output tiles does this wave own?
    %wid = func.call @wave_id() : () -> index
    %wave_m_off = layout.linearize %wid,
        #layout.strided_layout<[{{M_WAVES}}, {{N_WAVES}}] : [{{M_T}}, 0]>
    %wave_n_off = layout.linearize %wid,
        #layout.strided_layout<[{{M_WAVES}}, {{N_WAVES}}] : [0, {{N_T}}]>

    // Compose: tile index = WG offset + wave offset (for MFMA + C store)
    %m_base = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_m_off, %wave_m_off)
    %n_base = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_n_off, %wave_n_off)

    // Wave LOAD distribution: 2-D cooperative across M x K (and N x K).
    // waves_m * waves_k = NUM_WAVES. Each wave loads coop_m x coop_k tiles.
    // OOB waves clamp to last valid start in each dimension.
    %c_COOP_A_M = arith.constant {{COOP_A_M}} : index
    %c_COOP_A_K = arith.constant {{COOP_A_K}} : index
    %c_MAX_A_M = arith.constant {{MAX_COOP_A_M_START}} : index
    %c_MAX_A_K = arith.constant {{MAX_COOP_A_K_START}} : index
    // 2-D layout: wid -> (m_start, k_start) via [waves_m, waves_k]:[coop_m, coop_k]
    %coop_a_m_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_A_WAVES_M}}, {{COOP_A_WAVES_K}}] : [{{COOP_A_M}}, 0]>
    %coop_a_k_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_A_WAVES_M}}, {{COOP_A_WAVES_K}}] : [0, {{COOP_A_K}}]>
    %coop_a_m_start = arith.minui %coop_a_m_raw, %c_MAX_A_M : index
    %coop_a_k_start = arith.minui %coop_a_k_raw, %c_MAX_A_K : index
    %c_COOP_B_N = arith.constant {{COOP_B_N}} : index
    %c_COOP_B_K = arith.constant {{COOP_B_K}} : index
    %c_MAX_B_N = arith.constant {{MAX_COOP_B_N_START}} : index
    %c_MAX_B_K = arith.constant {{MAX_COOP_B_K_START}} : index
    %coop_b_n_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_B_WAVES_N}}, {{COOP_B_WAVES_K}}] : [{{COOP_B_N}}, 0]>
    %coop_b_k_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_B_WAVES_N}}, {{COOP_B_WAVES_K}}] : [0, {{COOP_B_K}}]>
    %coop_b_n_start = arith.minui %coop_b_n_raw, %c_MAX_B_N : index
    %coop_b_k_start = arith.minui %coop_b_k_raw, %c_MAX_B_K : index
    %coop_a_global = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_m_off, %coop_a_m_start)
    %coop_b_global = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_n_off, %coop_b_n_start)
    %tiles_per_slice_a = arith.constant {{A_TILES_PER_SLICE}} : index
    %tiles_per_slice_b = arith.constant {{B_TILES_PER_SLICE}} : index

    // === Initialize accumulators (constexpr over M_T*N_T) ===
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    // Address computations are decoupled from memory ops so they can be
    // repositioned to interleave with MFMA / hide latency.
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // Stage GLOBAL_LOAD: allocate LDS (1024 bytes per 16x32 tile).
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{A_STAGE_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{A_STAGE_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{A_STAGE_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{A_STAGE_LOAD}} : i32} : index

      // --- 2-D cooperative global load: each wave loads coop_m x coop_k tiles ---
      %k_a = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%k, %coop_a_k_start)
      %gfut_a = func.call @k_load_a_16x32_from_global(%c_COOP_A_M, %c_COOP_A_K, %A_rsrc, %k_a, %stride_AB, %coop_a_global)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_a_buf
      %k_b = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%k, %coop_b_k_start)
      %gfut_b = func.call @k_load_b_16x32_from_global(%c_COOP_B_N, %c_COOP_B_K, %B_rsrc, %k_b, %stride_AB, %coop_b_global)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_b_buf

      // --- 2-D cooperative LDS write: wave_base = k_start * tps + m_start ---
      %lds_wave_base_a = affine.apply affine_map<(m, k)[tps] -> (k * tps + m)>
          (%coop_a_m_start, %coop_a_k_start)[%tiles_per_slice_a]
      %lds_w_addrs_a = func.call @k_compute_lds_write_addrs_a(%c_COOP_A_M, %c_COOP_A_K, %base_a, %lds_wave_base_a, %tiles_per_slice_a)
          : (index, index, index, index, index) -> memref<?xindex>
      %lds_wave_base_b = affine.apply affine_map<(n, k)[tps] -> (k * tps + n)>
          (%coop_b_n_start, %coop_b_k_start)[%tiles_per_slice_b]
      %lds_w_addrs_b = func.call @k_compute_lds_write_addrs_b(%c_COOP_B_N, %c_COOP_B_K, %base_b, %lds_wave_base_b, %tiles_per_slice_b)
          : (index, index, index, index, index) -> memref<?xindex>

      // --- LDS read: each wave reads M_T/N_T tiles it needs for MFMA ---
      %lds_r_addrs_a = func.call @k_compute_lds_read_addrs_a(%c_M_T, %c_K_T, %base_a, %wave_m_off, %tiles_per_slice_a)
          : (index, index, index, index, index) -> memref<?xindex>
      %lds_r_addrs_b = func.call @k_compute_lds_read_addrs_b(%c_N_T, %c_K_T, %base_b, %wave_n_off, %tiles_per_slice_b)
          : (index, index, index, index, index) -> memref<?xindex>

      // --- Wait for cooperative loads + store to LDS ---
      %tok_a = func.call @k_store_to_lds_at_addrs_a(%lds_w_addrs_a, %c_COOP_A_M, %c_COOP_A_K, %gfut_a)
          : (memref<?xindex>, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_to_lds_at_addrs_b(%lds_w_addrs_b, %c_COOP_B_N, %c_COOP_B_K, %gfut_b)
          : (memref<?xindex>, index, index, !gfut_b_buf) -> !tok_b_buf

      // --- Wait LDS writes then barrier (waitcnt before barrier for visibility) ---
      func.call @k_wait_lds_writes(%tok_a)
          : (memref<?x!lds_write_token>) -> ()
      func.call @k_wait_lds_writes(%tok_b)
          : (memref<?x!lds_write_token>) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{A_STAGE_READ}} : i32}

      // --- Issue LDS reads at pre-computed addresses ---
      %a_fut = func.call @k_read_lds_at_addrs_a(%lds_r_addrs_a)
          : (memref<?xindex>) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_at_addrs_b(%lds_r_addrs_b)
          : (memref<?xindex>) -> !fut_b_buf

      // Stage COMPUTE: fused wait + MFMA (constexpr over M_T x (K_T*2) x N_T)
      %c_K_MFMA = affine.apply affine_map<()[k] -> (k * 2)>()[%c_K_T]
      func.call @k_wait_and_compute_mfmas(%c_M_T, %c_N_T, %c_K_MFMA, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf) -> ()

      // Deallocate LDS at COMPUTE stage.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{A_STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{A_STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_rsrc, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
