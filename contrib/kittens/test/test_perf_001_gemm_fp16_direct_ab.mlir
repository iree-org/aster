// Multi-workgroup multi-wave constexpr GEMM with direct-AB (zero LDS).
// Both A and B preshuffled on host, loaded directly into registers.
// A operand: buffer_load_dwordx4 -> split vx4 -> MFMA (no LDS).
// B operand: buffer_load_dwordx4 -> split vx4 -> MFMA (no LDS).
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

amdgcn.module @kittens_gemm_f16_direct_ab target = #amdgcn.target<gfx942> {
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

  // Multi-WG multi-wave GEMM: both A and B direct (zero LDS).
  amdgcn.kernel @gemm_f16_direct_ab arguments <[
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
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // WG tile offsets via layout
    %bid = func.call @linear_block_id() : () -> index
    %wg_m_off = layout.linearize %bid,
        #layout.strided_layout<[{{M_WG}}, {{N_WG}}] : [{{M_TILES_WG}}, 0]>
    %wg_n_off = layout.linearize %bid,
        #layout.strided_layout<[{{M_WG}}, {{N_WG}}] : [0, {{N_TILES_WG}}]>

    // Wave COMPUTE distribution (= LOAD distribution for direct path)
    %wid = func.call @wave_id() : () -> index
    %wave_m_off = layout.linearize %wid,
        #layout.strided_layout<[{{M_WAVES}}, {{N_WAVES}}] : [{{M_T}}, 0]>
    %wave_n_off = layout.linearize %wid,
        #layout.strided_layout<[{{M_WAVES}}, {{N_WAVES}}] : [0, {{N_T}}]>

    // Compose: tile index = WG offset + wave offset
    %m_base = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_m_off, %wave_m_off)
    %n_base = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%wg_n_off, %wave_n_off)

    // === Initialize accumulators ===
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop: both A and B direct (zero LDS) ===
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // --- A_STAGE_LOAD: issue A + B preshuffle loads ---
      %gfut_a = func.call @k_load_a_direct(%c_M_T, %c_K_T, %A_rsrc, %k, %m_base)
          : (index, index, !aster_utils.any, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_direct(%c_N_T, %c_K_T, %B_rsrc, %k, %n_base)
          : (index, index, !aster_utils.any, index, index) -> !gfut_b_buf

      // --- Wait A+B, split, MFMA (stages inside function) ---
      %c_K_MFMA = affine.apply affine_map<()[k] -> (k * 2)>()[%c_K_T]
      func.call @k_wait_split_compute_direct_ab(%c_M_T, %c_N_T, %c_K_T, %c_K_MFMA,
          %gfut_a, %gfut_b, %C_buf)
          : (index, index, index, index,
             !gfut_a_buf, !gfut_b_buf, !c_buf) -> ()
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_rsrc, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
