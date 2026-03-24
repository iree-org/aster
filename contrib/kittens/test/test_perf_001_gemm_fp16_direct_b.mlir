// Multi-workgroup multi-wave constexpr GEMM with direct-B (LDS bypass for B).
// A operand: global_load_dwordx4 -> LDS -> MFMA (standard path).
// B operand: buffer_load_dwordx4 -> split vx4 -> MFMA (no LDS, no bpermute).
// Uses 16x16x16 MFMA with dwordx4 global loads (16x32 transfer tiles).

// Register type aliases
!v   = !amdgcn.vgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
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

// Memref buffer type aliases
!gfut_a_buf = memref<?x!future_global_read>
!gfut_b_buf = memref<?x!future_global_read>
!tok_a_buf = memref<?x!lds_write_token>
!tok_b_buf = memref<?x!lds_write_token>
!fut_a_buf = memref<?x!future_lds_read>
!fut_b_buf = memref<?x!future_lds_read>
!vals_a_buf = memref<?x!rt_A_f16>
!vals_b_buf = memref<?x!rt_B_f16>
!c_buf = memref<?x!rt_C_f32>

amdgcn.module @kittens_gemm_f16_direct_b target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library functions
  func.func private @linear_block_id() -> index
  func.func private @wave_id() -> index
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !aster_utils.any, index, index, index)
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
  // From global_16x64_b[_buf].mlir -- type-erases raw !sx2 arg to !aster_utils.any
  func.func private @prepare_ptr(!sx2) -> !aster_utils.any

{{K_LOOP_HELPERS}}

  // Multi-WG multi-wave GEMM: A via LDS, B via bpermute (no LDS).
  amdgcn.kernel @gemm_f16_direct_b arguments <[
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_K_T = arith.constant {{K_T}} : index
    %c_M_TILES_WG = arith.constant {{M_TILES_WG}} : index
    %c_N_TILES_WG = arith.constant {{N_TILES_WG}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index
    // Preshuffle B stride: (K / 32) * 1024 bytes.
    %stride_n0_bytes = arith.constant {{STRIDE_N0_BYTES}} : index

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

    // Wave LOAD distribution: 2-D cooperative for A (LDS), per-wave for B (direct).
    // OOB waves clamp to last valid start in each dimension.
    %c_COOP_A_M = arith.constant {{COOP_A_M}} : index
    %c_COOP_A_K = arith.constant {{COOP_A_K}} : index
    %c_MAX_A_M = arith.constant {{MAX_COOP_A_M_START}} : index
    %c_MAX_A_K = arith.constant {{MAX_COOP_A_K_START}} : index
    %coop_a_m_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_A_WAVES_M}}, {{COOP_A_WAVES_K}}] : [{{COOP_A_M}}, 0]>
    %coop_a_k_raw = layout.linearize %wid,
        #layout.strided_layout<[{{COOP_A_WAVES_M}}, {{COOP_A_WAVES_K}}] : [0, {{COOP_A_K}}]>
    %coop_a_m_start = arith.minui %coop_a_m_raw, %c_MAX_A_M : index
    %coop_a_k_start = arith.minui %coop_a_k_raw, %c_MAX_A_K : index
    %coop_a_global = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_m_off, %coop_a_m_start)
    %tiles_per_slice_a = arith.constant {{A_TILES_PER_SLICE}} : index

    // === Initialize accumulators ===
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop: A via LDS, B direct (preshuffled, no LDS) ===
    // Pipeline: A and B global loads at A_STAGE_LOAD,
    //   A LDS write + B wait/split at A_STAGE_WRITE,
    //   A LDS read at A_STAGE_READ,
    //   MFMA at A_STAGE_COMPUTE.
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // LDS only for A.
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{A_STAGE_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{A_STAGE_LOAD}} : i32} : index

      // --- 2-D cooperative A global load + per-wave B direct load ---
      %k_a = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%k, %coop_a_k_start)
      %gfut_a = func.call @k_load_a_16x32_from_global(%c_COOP_A_M, %c_COOP_A_K, %A_rsrc, %k_a, %stride_AB, %coop_a_global)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_direct(%c_N_T, %c_K_T, %B_rsrc, %k, %n_base)
          : (index, index, !aster_utils.any, index, index) -> !gfut_b_buf

      // --- 2-D cooperative A LDS write ---
      %lds_wave_base_a = affine.apply affine_map<(m, k)[tps] -> (k * tps + m)>
          (%coop_a_m_start, %coop_a_k_start)[%tiles_per_slice_a]
      %lds_w_addrs_a = func.call @k_compute_lds_write_addrs_a(%c_COOP_A_M, %c_COOP_A_K, %base_a, %lds_wave_base_a, %tiles_per_slice_a)
          : (index, index, index, index, index) -> memref<?xindex>
      // --- A LDS read: each wave reads M_T x K_T tiles for MFMA ---
      %lds_r_addrs_a = func.call @k_compute_lds_read_addrs_a(%c_M_T, %c_K_T, %base_a, %wave_m_off, %tiles_per_slice_a)
          : (index, index, index, index, index) -> memref<?xindex>
      %tok_a = func.call @k_store_to_lds_at_addrs_a(%lds_w_addrs_a, %c_COOP_A_M, %c_COOP_A_K, %gfut_a)
          : (memref<?xindex>, index, index, !gfut_a_buf) -> !tok_a_buf

      // --- Wait A LDS writes then barrier (waitcnt before barrier for visibility) ---
      func.call @k_wait_lds_writes(%tok_a)
          : (memref<?x!lds_write_token>) -> ()
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{A_STAGE_READ}} : i32}
      %a_fut = func.call @k_read_lds_at_addrs_a(%lds_r_addrs_a)
          : (memref<?xindex>) -> !fut_a_buf

      // --- A_STAGE_COMPUTE: B wait+split+MFMA (B wait annotated DS_WRITE, MFMA at COMPUTE) ---
      %c_K_MFMA = affine.apply affine_map<()[k] -> (k * 2)>()[%c_K_T]
      func.call @k_wait_split_compute_direct_b(%c_M_T, %c_N_T, %c_K_T, %c_K_MFMA,
          %gfut_b, %a_fut, %C_buf)
          : (index, index, index, index,
             !gfut_b_buf, !fut_a_buf, !c_buf) -> ()

      // Deallocate A LDS.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{A_STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_rsrc, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
