// Multi-workgroup multi-wave constexpr GEMM with LDS + pipelining:
// C[M_DIM x N_DIM] = A[M_DIM x K] @ B[N_DIM x K]^T
//
// Two-level parallelism:
//   Workgroup grid: WG_M x WG_N workgroups, num_blocks = WG_M * WG_N
//   Wave grid:      M_WAVES x N_WAVES waves per WG
//   M_DIM = WG_M * M_WAVES * M_T * 16
//   N_DIM = WG_N * N_WAVES * N_T * 16
//
// Each WG has M_WAVES * N_WAVES waves (threads = that * 64).
// wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
// Wave (wm, wn) within WG (wg_m, wg_n) computes M_T x N_T tiles at:
//   m_base = (wg_m * M_WAVES + wave_m) * M_T  (tile units)
//   n_base = (wg_n * N_WAVES + wave_n) * N_T  (tile units)
//
// LDS layout per K-iteration (single large allocations):
//   A: A_LDS_BYTES = M_WAVES * M_T * 512 bytes
//   B: B_LDS_BYTES = N_WAVES * N_T * 512 bytes
//   Per-wave offsets computed arithmetically from base.
//   Total: (A_LDS_BYTES + B_LDS_BYTES) per pipeline stage.
//
// Cross-wave barrier (s_barrier) between DS_WRITE and DS_READ ensures all
// waves' LDS writes are visible before any wave reads.
//
// Scalar substitutions:
//   M_T, N_T, MN             - per-wave tile dimensions and product
//   WG_M, WG_N               - workgroup grid dimensions
//   M_WAVES, N_WAVES  - wave grid dimensions within each WG
//   A_LDS_BYTES, B_LDS_BYTES  - LDS allocation sizes
//   K, K_TILES, STRIDE_AB, STRIDE_C, SHARED_MEM
//   STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE

// Register type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
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
// Both A and B use single large LDS allocations with arithmetic offsets.
!gfut_a_buf = memref<{{M_T}} x !future_global_read>
!gfut_b_buf = memref<{{N_T}} x !future_global_read>
!tok_a_buf = memref<{{M_T}} x !lds_write_token>
!tok_b_buf = memref<{{N_T}} x !lds_write_token>
!fut_a_buf = memref<{{M_T}} x !future_lds_read>
!fut_b_buf = memref<{{N_T}} x !future_lds_read>
!vals_a_buf = memref<{{M_T}} x !rt_A_f16>
!vals_b_buf = memref<{{N_T}} x !rt_B_f16>
!c_buf = memref<{{MN}} x !rt_C_f32>

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

  // Issue global loads for A tiles (no wait, returns futures).
  // m_base: WG's starting tile index in M (= wg_id * M_T).
  func.func private @k_load_a_from_global(%m_t: index,
      %A_ptr: !sx2, %k_offset: index, %stride_AB: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %gfut_a = memref.alloca() : !gfut_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%m_base]
      %fut = func.call @load_global_tile_f16(%A_ptr, %m_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      memref.store %fut, %gfut_a[%i] : !gfut_a_buf
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads for B tiles (no wait, returns futures).
  // n_base: WG's starting tile index in N (= wg_n * N_T).
  func.func private @k_load_b_from_global(%n_t: index,
      %B_ptr: !sx2, %k_offset: index, %stride_AB: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %gfut_b = memref.alloca() : !gfut_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%n_base]
      %fut = func.call @load_global_tile_f16(%B_ptr, %n_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      memref.store %fut, %gfut_b[%i] : !gfut_b_buf
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Wait for A global loads and store to LDS with xor swizzle.
  // A offsets computed arithmetically: base_a + (wave_a_base + i) * 512.
  func.func private @k_store_a_to_lds(%m_t: index,
      %base_a: index, %wave_a_base: index, %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_a = memref.alloca() : !tok_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %off = affine.apply affine_map<(i)[base, wab] -> (base + (wab + i) * 512)>(%i)[%base_a, %wave_a_base]
      %gfut = memref.load %gfut_a[%i] : !gfut_a_buf
      %tok = func.call @store_global_tile_to_lds_xor_swizzle_f16(%off, %gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> !lds_write_token
      memref.store %tok, %tok_a[%i] : !tok_a_buf
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Wait for B global loads and store to LDS with xor swizzle.
  // B offsets computed arithmetically: base_b + (wave_b_base + i) * 512.
  func.func private @k_store_b_to_lds(%n_t: index,
      %base_b: index, %wave_b_base: index, %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_b = memref.alloca() : !tok_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %off = affine.apply affine_map<(i)[base, wbb] -> (base + (wbb + i) * 512)>(%i)[%base_b, %wave_b_base]
      %gfut = memref.load %gfut_b[%i] : !gfut_b_buf
      %tok = func.call @store_global_tile_to_lds_xor_swizzle_f16(%off, %gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> !lds_write_token
      memref.store %tok, %tok_b[%i] : !tok_b_buf
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Wait for A LDS writes, cross-wave barrier, then load A tiles from LDS.
  // The s_barrier ensures all waves' LDS writes (both A and B) are visible
  // before any wave reads. This is safe because k_store_b_to_lds has already
  // been called before this function (B write tokens are consumed separately).
  // A offsets computed arithmetically: base_a + (wave_a_base + i) * 512.
  func.func private @k_sync_and_read_lds_a(%m_t: index,
      %tok_a: !tok_a_buf, %base_a: index, %wave_a_base: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %m_t step %c1 {
      %tok = memref.load %tok_a[%i] : !tok_a_buf
      amdgcn.wait deps %tok {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
    } {aster.constexpr}
    // Cross-wave barrier: all M_WAVES waves must complete LDS writes
    // before any wave reads (B tiles are shared across waves).
    amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32}
    %a_fut = memref.alloca() : !fut_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %off = affine.apply affine_map<(i)[base, wab] -> (base + (wab + i) * 512)>(%i)[%base_a, %wave_a_base]
      %fut = func.call @load_lds_A_xor_swizzle_f16(%off)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %a_fut[%i] : !fut_a_buf
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Wait for B LDS writes then load B tiles from LDS into register futures.
  // B offsets computed arithmetically: base_b + (wave_b_base + i) * 512.
  func.func private @k_sync_and_read_lds_b(%n_t: index,
      %tok_b: !tok_b_buf, %base_b: index, %wave_b_base: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n_t step %c1 {
      %tok = memref.load %tok_b[%i] : !tok_b_buf
      amdgcn.wait deps %tok {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
    } {aster.constexpr}
    %b_fut = memref.alloca() : !fut_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %off = affine.apply affine_map<(i)[base, wbb] -> (base + (wbb + i) * 512)>(%i)[%base_b, %wave_b_base]
      %fut = func.call @load_lds_B_xor_swizzle_f16(%off)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %b_fut[%i] : !fut_b_buf
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Extract A register values from LDS read futures.
  func.func private @k_extract_lds_values_a(%m_t: index, %a_fut: !fut_a_buf) -> !vals_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a_vals = memref.alloca() : !vals_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %fut = memref.load %a_fut[%i] : !fut_a_buf
      %a = func.call @get_lds_A_f16(%fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      memref.store %a, %a_vals[%i] : !vals_a_buf
    } {aster.constexpr}
    return %a_vals : !vals_a_buf
  }

  // Extract B register values from LDS read futures.
  func.func private @k_extract_lds_values_b(%n_t: index, %b_fut: !fut_b_buf) -> !vals_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b_vals = memref.alloca() : !vals_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %fut = memref.load %b_fut[%i] : !fut_b_buf
      %b = func.call @get_lds_B_f16(%fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      memref.store %b, %b_vals[%i] : !vals_b_buf
    } {aster.constexpr}
    return %b_vals : !vals_b_buf
  }

  // Compute MFMAs: accumulate A*B into C tiles.
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index,
      %a_vals: !vals_a_buf, %b_vals: !vals_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %a = memref.load %a_vals[%mt] : !vals_a_buf
        %b = memref.load %b_vals[%nt] : !vals_b_buf
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
        %c_old = memref.load %c_buf[%idx] : !c_buf
        %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
        memref.store %c_new, %c_buf[%idx] : !c_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Store C accumulator tiles to global memory.
  // m_base/n_base: WG's starting tile indices (= wg_m * M_T, wg_n * N_T).
  func.func private @store_c_tiles(%m_t: index, %n_t: index,
      %c_buf: !c_buf, %C_ptr: !sx2, %stride_C: index,
      %m_base: index, %n_base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
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
  // num_blocks = WG_M * WG_N; flat block ID delinearized into (wg_m, wg_n).
  // wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
  amdgcn.kernel @gemm_f16_weak_scaled arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{SHARED_MEM}} : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_MN  = arith.constant {{MN}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // Delinearize flat block ID into (wg_m, wg_n) workgroup coordinates.
    %flat_id = gpu.block_id x
    %c_WG_M = arith.constant {{WG_M}} : index
    %c_WG_N = arith.constant {{WG_N}} : index
    %wg_m, %wg_n = affine.delinearize_index %flat_id into (%c_WG_M, %c_WG_N) : index, index

    // Wave position within WG: delinearize wave_id into (wave_m, wave_n)
    %wid = func.call @wave_id() : () -> index
    %c_M_WAVES = arith.constant {{M_WAVES}} : index
    %c_N_WAVES = arith.constant {{N_WAVES}} : index
    %wave_m, %wave_n = affine.delinearize_index %wid into (%c_M_WAVES, %c_N_WAVES) : index, index

    // m_base = (wg_m * M_WAVES + wave_m) * M_T  (tile units, for global A/C addressing)
    // n_base = (wg_n * N_WAVES + wave_n) * N_T  (tile units, for global B/C addressing)
    // wave_a_base = wave_m * M_T                     (tile index, for LDS A offset arithmetic)
    // wave_b_base = wave_n * N_T                     (tile index, for LDS B offset arithmetic)
    %m_base = affine.apply affine_map<(wgm, wm)[mt, nw] -> ((wgm * nw + wm) * mt)>
        (%wg_m, %wave_m)[%c_M_T, %c_M_WAVES]
    %n_base = affine.apply affine_map<(wgn, wn)[nt, nw] -> ((wgn * nw + wn) * nt)>
        (%wg_n, %wave_n)[%c_N_T, %c_N_WAVES]
    %wave_a_base = affine.apply affine_map<(wm)[mt] -> (wm * mt)>(%wave_m)[%c_M_T]
    %wave_b_base = affine.apply affine_map<(wn)[nt] -> (wn * nt)>(%wave_n)[%c_N_T]

    // === Initialize accumulators (constexpr over M_T*N_T) ===
    // Stored in memref -- promote-loop-carried-memrefs converts to iter_args.
    %C_buf = memref.alloca() : !c_buf
    scf.for %i = %c0 to %c_MN step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    scf.for %k = %c0 to %K_tiles step %c1 {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Stage GLOBAL_LOAD: allocate LDS.
      // A: single block of A_LDS_BYTES = M_WAVES * M_T * 512.
      // B: single block of B_LDS_BYTES = N_WAVES * N_T * 512.
      %lds_a_h = amdgcn.alloc_lds {{A_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_a = amdgcn.get_lds_offset %lds_a_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index
      %lds_b_h = amdgcn.alloc_lds {{B_LDS_BYTES}} {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %base_b = amdgcn.get_lds_offset %lds_b_h {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : index

      // Issue global loads (each wave loads its own A tiles + shared B tiles).
      %gfut_a = func.call @k_load_a_from_global(%c_M_T, %A_ptr, %k_offset, %stride_AB, %m_base)
          : (index, !sx2, index, index, index) -> !gfut_a_buf
      %gfut_b = func.call @k_load_b_from_global(%c_N_T, %B_ptr, %k_offset, %stride_AB, %n_base)
          : (index, !sx2, index, index, index) -> !gfut_b_buf

      // Stage DS_WRITE: wait for globals, store to LDS.
      // Each wave stores M_T A tiles at base_a + (wave_a_base + i) * 512,
      // and N_T B tiles at base_b + (wave_b_base + i) * 512.
      %tok_a = func.call @k_store_a_to_lds(%c_M_T, %base_a, %wave_a_base, %gfut_a)
          : (index, index, index, !gfut_a_buf) -> !tok_a_buf
      %tok_b = func.call @k_store_b_to_lds(%c_N_T, %base_b, %wave_b_base, %gfut_b)
          : (index, index, index, !gfut_b_buf) -> !tok_b_buf

      // Stage DS_READ: wait for this wave's LDS writes + cross-wave barrier,
      // then load from LDS. The s_barrier (inside k_sync_and_read_lds_a)
      // ensures all waves' writes are visible before any wave reads.
      %a_fut = func.call @k_sync_and_read_lds_a(%c_M_T, %tok_a, %base_a, %wave_a_base)
          : (index, !tok_a_buf, index, index) -> !fut_a_buf
      %b_fut = func.call @k_sync_and_read_lds_b(%c_N_T, %tok_b, %base_b, %wave_b_base)
          : (index, !tok_b_buf, index, index) -> !fut_b_buf

      // Stage COMPUTE: extract register values from futures
      %a_vals = func.call @k_extract_lds_values_a(%c_M_T, %a_fut)
          : (index, !fut_a_buf) -> !vals_a_buf
      %b_vals = func.call @k_extract_lds_values_b(%c_N_T, %b_fut)
          : (index, !fut_b_buf) -> !vals_b_buf

      // Stage COMPUTE: MFMAs (constexpr over M_T x N_T)
      func.call @k_compute_mfmas(%c_M_T, %c_N_T, %a_vals, %b_vals, %C_buf)
          : (index, index, !vals_a_buf, !vals_b_buf, !c_buf) -> ()

      // Stage COMPUTE: deallocate LDS (single A block + single B block)
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_ptr, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
