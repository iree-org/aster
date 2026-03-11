// Multi-workgroup multi-wave constexpr GEMM with direct-A (LDS bypass for A).
// A operand: global_load_dwordx4 -> ds_bpermute_b32 -> MFMA (no LDS allocation).
// B operand: global_load_dwordx4 -> LDS -> MFMA (unchanged from perf_001).
// Uses 16x16x16 MFMA with dwordx4 global loads (16x32 transfer tiles).

// Register type aliases
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

amdgcn.module @kittens_gemm_f16_direct_a target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library functions
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

  // Multi-WG multi-wave GEMM: A via bpermute (no LDS), B via LDS.
  amdgcn.kernel @gemm_f16_direct_a arguments <[
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
    %tiles_per_slice_b = arith.constant {{B_TILES_PER_SLICE}} : index

    // Delinearize flat block ID into (m_wg, n_wg) workgroup coordinates.
    %flat_id = gpu.block_id x
    %c_M_WG = arith.constant {{M_WG}} : index
    %c_N_WG = arith.constant {{N_WG}} : index
    %m_wg, %n_wg = affine.delinearize_index %flat_id into (%c_M_WG, %c_N_WG) : index, index

    // Wave position within WG
    %wid = func.call @wave_id() : () -> index
    %c_M_WAVES = arith.constant {{M_WAVES}} : index
    %c_N_WAVES = arith.constant {{N_WAVES}} : index
    %wave_m, %wave_n = affine.delinearize_index %wid into (%c_M_WAVES, %c_N_WAVES) : index, index

    %m_base = affine.apply affine_map<(mwg, wm)[mt_wg, mt] -> (mwg * mt_wg + wm * mt)>
        (%m_wg, %wave_m)[%c_M_TILES_WG, %c_M_T]
    %n_base = affine.apply affine_map<(nwg, wn)[nt_wg, nt] -> (nwg * nt_wg + wn * nt)>
        (%n_wg, %wave_n)[%c_N_TILES_WG, %c_N_T]
    %wave_b_base = affine.apply affine_map<(wn)[nt] -> (wn * nt)>(%wave_n)[%c_N_T]

    // === Initialize accumulators ===
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop: A via bpermute, B via LDS ===
    // Address computations are decoupled from memory ops so they can be
    // repositioned to interleave with MFMA / hide latency.
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // LDS only for B (A bypasses LDS entirely).
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // Issue dwordx4 global loads for both A and B.
      %gfut_a = func.call @k_load_a_16x32_from_global(%c_M_T, %c_K_T, %A_rsrc, %k, %stride_AB, %m_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_16x32_from_global(%c_N_T, %c_K_T, %B_rsrc, %k, %stride_AB, %n_base)
          : (index, index, !aster_utils.any, index, index, index) -> !gfut_b_buf

      // --- Compute B LDS write addresses (overlaps with global load latency) ---
      %lds_w_addrs_b = func.call @k_compute_lds_write_addrs_b(%c_N_T, %c_K_T, %base_b, %wave_b_base, %tiles_per_slice_b)
          : (index, index, index, index, index) -> memref<?xindex>

      // --- Compute B LDS read addresses (overlaps with global load latency) ---
      %lds_r_addrs_b = func.call @k_compute_lds_read_addrs_b(%c_N_T, %c_K_T, %base_b, %wave_b_base, %tiles_per_slice_b)
          : (index, index, index, index, index) -> memref<?xindex>

      // --- Wait for B global loads + store to LDS at pre-computed addresses ---
      %tok_b = func.call @k_store_to_lds_at_addrs_b(%lds_w_addrs_b, %c_N_T, %c_K_T, %gfut_b)
          : (memref<?xindex>, index, index, !gfut_b_buf) -> !tok_b_buf

      // --- Barrier then wait B LDS write tokens ---
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32}
      func.call @k_wait_lds_writes(%tok_b)
          : (memref<?x!lds_write_token>) -> ()

      // --- Issue B LDS reads at pre-computed addresses ---
      %b_fut = func.call @k_read_lds_at_addrs_b(%lds_r_addrs_b)
          : (memref<?xindex>) -> !fut_b_buf

      // A: extract directly from global loads via bpermute (no LDS).
      %a_vals = func.call @k_extract_direct_a_values(%c_M_T, %c_K_T, %gfut_a)
          : (index, index, !gfut_a_buf) -> !vals_a_buf

      // B: extract from LDS read futures.
      %b_vals = func.call @k_extract_lds_values_b(%b_fut)
          : (!fut_b_buf) -> !vals_b_buf

      // MFMA compute.
      %c_K_MFMA = affine.apply affine_map<()[k] -> (k * 2)>()[%c_K_T]
      func.call @k_compute_mfmas(%c_M_T, %c_N_T, %c_K_MFMA, %a_vals, %b_vals, %C_buf)
          : (index, index, index, !vals_a_buf, !vals_b_buf, !c_buf) -> ()

      // Deallocate B LDS.
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_rsrc, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
