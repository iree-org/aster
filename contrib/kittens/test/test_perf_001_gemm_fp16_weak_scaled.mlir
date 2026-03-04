// Multi-workgroup multi-wave constexpr GEMM with LDS + pipelining:
// C[M_DIM x N_DIM] = A[M_DIM x K] @ B[N_DIM x K]^T
//
// Two-level parallelism:
//   Workgroup grid: M_WG x N_WG workgroups, num_blocks = M_WG * N_WG
//   Wave grid:      M_WAVES x N_WAVES waves per WG
//   M_DIM = M_WG * M_WAVES * M_T * 16
//   N_DIM = N_WG * N_WAVES * N_T * 16
//
// Each WG has M_WAVES * N_WAVES waves (threads = that * 64).
// wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
// Wave (wm, wn) within WG (m_wg, n_wg) computes M_T x N_T tiles at:
//   m_base = (m_wg * M_WAVES + wave_m) * M_T  (tile units)
//   n_base = (n_wg * N_WAVES + wave_n) * N_T  (tile units)
//
// LDS layout per K-iteration (single large allocations, K_T slices):
//   A: A_LDS_BYTES = M_WAVES * M_T * K_T * 512 bytes
//   B: B_LDS_BYTES = N_WAVES * K_T * N_T * 512 bytes
//   Per-wave offsets: base + (kt * tiles_per_slice + wave_base + i) * 512.
//   Total: (A_LDS_BYTES + B_LDS_BYTES) per pipeline stage.
//
// Cross-wave barrier (s_barrier) between DS_WRITE and DS_READ ensures all
// waves' LDS writes are visible before any wave reads.
//
// Scalar substitutions:
//   M_T, N_T, K_T            - per-wave number of tiles
//   K_TILES                  - total K-tiles (K/16), K-loop steps by K_T
//   A_TILES_PER_SLICE, B_TILES_PER_SLICE - tiles per kt-slice in LDS
//   M_WG, N_WG               - workgroup grid dimensions
//   M_WAVES, N_WAVES         - wave grid dimensions within each WG
//   A_LDS_BYTES, B_LDS_BYTES - LDS allocation sizes
//   STRIDE_AB, STRIDE_C, SHARED_MEM
//   STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE

// Register type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

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

amdgcn.module @kittens_gemm_f16_weak_scaled target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library functions (external, provided by preload library)
  func.func private @wave_id() -> index
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token
  func.func private @load_global_tile_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_xor_swizzle_f16(index, !future_global_read) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // === K-loop helper functions (inlined before constexpr expansion) ===
  // When k_t > 1, each pipeline stage processes multiple K-tiles.
  // Buffers are linearized: index = linearize_index [kt, i] by (k_t, dim_t).

  // Issue global loads for A tiles across k_t K-tiles (no wait, returns futures).
  // %k: starting K-tile index for this pipeline stage.
  // m_base: WG's starting tile index in M (= wg_id * M_T).
  func.func private @k_load_a_from_global(%m_t: index, %k_t: index,
      %A_ptr: !sx2, %k: index, %stride_AB: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 16)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%m_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %fut = func.call @load_global_tile_f16(%A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> !future_global_read
        memref.store %fut, %gfut_a[%idx] : !gfut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads for B tiles across k_t K-tiles (no wait, returns futures).
  // %k: starting K-tile index for this pipeline stage.
  // n_base: WG's starting tile index in N (= n_wg * N_T).
  func.func private @k_load_b_from_global(%n_t: index, %k_t: index,
      %B_ptr: !sx2, %k: index, %stride_AB: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 16)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%n_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %fut = func.call @load_global_tile_f16(%B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> !future_global_read
        memref.store %fut, %gfut_b[%idx] : !gfut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Wait for A global loads and store to LDS with xor swizzle.
  // LDS layout: k_t slices, each with tiles_per_slice tiles.
  // Offset = base + (kt * tiles_per_slice + wave_base + i) * 512.
  func.func private @k_store_a_to_lds(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index,
      %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 512)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gfut = memref.load %gfut_a[%idx] : !gfut_a_buf
        %tok = func.call @store_global_tile_to_lds_xor_swizzle_f16(%off, %gfut)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !future_global_read) -> !lds_write_token
        memref.store %tok, %tok_a[%idx] : !tok_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Wait for B global loads and store to LDS with xor swizzle.
  // LDS layout: k_t slices, each with tiles_per_slice tiles.
  // Offset = base + (kt * tiles_per_slice + wave_base + i) * 512.
  func.func private @k_store_b_to_lds(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index,
      %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 512)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %gfut = memref.load %gfut_b[%idx] : !gfut_b_buf
        %tok = func.call @store_global_tile_to_lds_xor_swizzle_f16(%off, %gfut)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !future_global_read) -> !lds_write_token
        memref.store %tok, %tok_b[%idx] : !tok_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Wait for all A LDS write tokens (k_t * m_t tokens).
  func.func private @k_wait_lds_writes_a(%m_t: index, %k_t: index,
      %tok_a: !tok_a_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %tok = memref.load %tok_a[%idx] : !tok_a_buf
        amdgcn.wait deps %tok {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Wait for all B LDS write tokens (k_t * n_t tokens).
  func.func private @k_wait_lds_writes_b(%n_t: index, %k_t: index,
      %tok_b: !tok_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tok = memref.load %tok_b[%idx] : !tok_b_buf
        amdgcn.wait deps %tok {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Read A tiles from LDS (k_t * m_t tiles, with tiles_per_slice layout).
  func.func private @k_read_lds_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 512)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %fut = func.call @load_lds_A_xor_swizzle_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> !future_lds_read
        memref.store %fut, %a_fut[%idx] : !fut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS (k_t * n_t tiles, with tiles_per_slice layout).
  func.func private @k_read_lds_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 512)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %fut = func.call @load_lds_B_xor_swizzle_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> !future_lds_read
        memref.store %fut, %b_fut[%idx] : !fut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Extract A register values from LDS read futures (k_t * m_t tiles).
  func.func private @k_extract_lds_values_a(%m_t: index, %k_t: index,
      %a_fut: !fut_a_buf) -> !vals_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %a_vals = memref.alloca(%buf_size) : !vals_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %fut = memref.load %a_fut[%idx] : !fut_a_buf
        %a = func.call @get_lds_A_f16(%fut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_lds_read) -> !rt_A_f16
        memref.store %a, %a_vals[%idx] : !vals_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_vals : !vals_a_buf
  }

  // Extract B register values from LDS read futures (k_t * n_t tiles).
  func.func private @k_extract_lds_values_b(%n_t: index, %k_t: index,
      %b_fut: !fut_b_buf) -> !vals_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %b_vals = memref.alloca(%buf_size) : !vals_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %fut = memref.load %b_fut[%idx] : !fut_b_buf
        %b = func.call @get_lds_B_f16(%fut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_lds_read) -> !rt_B_f16
        memref.store %b, %b_vals[%idx] : !vals_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_vals : !vals_b_buf
  }

  // Compute MFMAs: single flat loop over m_t * k_t * n_t with delinearize.
  // Delinearize order (m, k, n) controls schedule: n varies fastest.
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index, %k_t: index,
      %a_vals: !vals_a_buf, %b_vals: !vals_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_t, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %kt, %nt = affine.delinearize_index %idx into (%m_t, %k_t, %n_t) : index, index, index
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %a_idx = affine.linearize_index [%kt, %mt] by (%k_t, %m_t) : index
      %b_idx = affine.linearize_index [%kt, %nt] by (%k_t, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %a = memref.load %a_vals[%a_idx] : !vals_a_buf
      %b = memref.load %b_vals[%b_idx] : !vals_b_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }

  // Store C accumulator tiles to global memory.
  // m_base/n_base: WG's starting tile indices (= m_wg * M_T, n_wg * N_T).
  func.func private @store_c_tiles(%m_t: index, %n_t: index,
      %c_buf: !c_buf, %C_ptr: !sx2, %stride_C: index,
      %m_base: index, %n_base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
        %c_tile = memref.load %c_buf[%idx] : !c_buf
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%mt)[%m_base]
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%nt)[%n_base]
        %tok = func.call @store_C_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !write_token
        amdgcn.wait deps %tok : !write_token
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Multi-WG multi-wave GEMM with pipelined LDS
  // M_WAVES * N_WAVES waves per WG; block_dim = (M_WAVES * N_WAVES * 64, 1, 1).
  // num_blocks = M_WG * N_WG; flat block ID delinearized into (m_wg, n_wg).
  // wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
  amdgcn.kernel @gemm_f16_weak_scaled arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{SHARED_MEM}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_K_T = arith.constant {{K_T}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index
    %tiles_per_slice_a = arith.constant {{A_TILES_PER_SLICE}} : index
    %tiles_per_slice_b = arith.constant {{B_TILES_PER_SLICE}} : index

    // Delinearize flat block ID into (m_wg, n_wg) workgroup coordinates.
    %flat_id = gpu.block_id x
    %c_M_WG = arith.constant {{M_WG}} : index
    %c_N_WG = arith.constant {{N_WG}} : index
    %m_wg, %n_wg = affine.delinearize_index %flat_id into (%c_M_WG, %c_N_WG) : index, index

    // Wave position within WG: delinearize wave_id into (wave_m, wave_n)
    %wid = func.call @wave_id() : () -> index
    %c_M_WAVES = arith.constant {{M_WAVES}} : index
    %c_N_WAVES = arith.constant {{N_WAVES}} : index
    %wave_m, %wave_n = affine.delinearize_index %wid into (%c_M_WAVES, %c_N_WAVES) : index, index

    // m_base = (m_wg * M_WAVES + wave_m) * M_T  (tile units, for global A/C addressing)
    // n_base = (n_wg * N_WAVES + wave_n) * N_T  (tile units, for global B/C addressing)
    // wave_a_base = wave_m * M_T                     (tile index, for LDS A offset arithmetic)
    // wave_b_base = wave_n * N_T                     (tile index, for LDS B offset arithmetic)
    %m_base = affine.apply affine_map<(mwg, wm)[mt, nw] -> ((mwg * nw + wm) * mt)>
        (%m_wg, %wave_m)[%c_M_T, %c_M_WAVES]
    %n_base = affine.apply affine_map<(nwg, wn)[nt, nw] -> ((nwg * nw + wn) * nt)>
        (%n_wg, %wave_n)[%c_N_T, %c_N_WAVES]
    %wave_a_base = affine.apply affine_map<(wm)[mt] -> (wm * mt)>(%wave_m)[%c_M_T]
    %wave_b_base = affine.apply affine_map<(wn)[nt] -> (wn * nt)>(%wave_n)[%c_N_T]

    // === Initialize accumulators (constexpr over M_T*N_T) ===
    // Stored in memref -- promote-loop-carried-memrefs converts to iter_args.
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    // Each iteration processes K_T K-tiles. Loop steps by K_T over K_TILES.
    // %k is directly the starting tile index (k_base_tiles).
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // Stage GLOBAL_LOAD: allocate LDS.
      // A: A_LDS_BYTES = M_WAVES * M_T * K_T * 512.
      // B: B_LDS_BYTES = N_WAVES * N_T * K_T * 512.
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // Issue global loads for K_T batches of tiles.
      %gfut_a = func.call @k_load_a_from_global(%c_M_T, %c_K_T, %A_ptr, %k, %stride_AB, %m_base)
          : (index, index, !sx2, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_from_global(%c_N_T, %c_K_T, %B_ptr, %k, %stride_AB, %n_base)
          : (index, index, !sx2, index, index, index) -> !gfut_b_buf

      // Stage DS_WRITE: store K_T batches to LDS.
      %tok_a = func.call @k_store_a_to_lds(%c_M_T, %c_K_T, %base_a, %wave_a_base, %tiles_per_slice_a, %gfut_a)
          : (index, index, index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_to_lds(%c_N_T, %c_K_T, %base_b, %wave_b_base, %tiles_per_slice_b, %gfut_b)
          : (index, index, index, index, index, !gfut_b_buf) -> !tok_b_buf

      // Stage DS_READ: wait all tokens, barrier, read all tiles.
      func.call @k_wait_lds_writes_a(%c_M_T, %c_K_T, %tok_a)
          : (index, index, !tok_a_buf) -> ()
      func.call @k_wait_lds_writes_b(%c_N_T, %c_K_T, %tok_b)
          : (index, index, !tok_b_buf) -> ()
      // Cross-wave barrier: all waves must complete LDS writes before any reads.
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32}
      %a_fut = func.call @k_read_lds_a(%c_M_T, %c_K_T, %base_a, %wave_a_base, %tiles_per_slice_a)
          : (index, index, index, index, index) -> !fut_a_buf
      %b_fut = func.call @k_read_lds_b(%c_N_T, %c_K_T, %base_b, %wave_b_base, %tiles_per_slice_b)
          : (index, index, index, index, index) -> !fut_b_buf

      // Stage COMPUTE: extract register values from futures
      %a_vals = func.call @k_extract_lds_values_a(%c_M_T, %c_K_T, %a_fut)
          : (index, index, !fut_a_buf) -> !vals_a_buf
      %b_vals = func.call @k_extract_lds_values_b(%c_N_T, %c_K_T, %b_fut)
          : (index, index, !fut_b_buf) -> !vals_b_buf

      // Stage COMPUTE: MFMAs (constexpr over K_T x M_T x N_T)
      func.call @k_compute_mfmas(%c_M_T, %c_N_T, %c_K_T, %a_vals, %b_vals, %C_buf)
          : (index, index, index, !vals_a_buf, !vals_b_buf, !c_buf) -> ()

      // Stage COMPUTE: deallocate LDS
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_ptr, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
