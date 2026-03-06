// Multi-workgroup multi-wave constexpr GEMM with LDS + pipelining (32x32x8 MFMA):
// C[M_DIM x N_DIM] = A[M_DIM x K] @ B[N_DIM x K]^T
//
// Uses v_mfma_f32_32x32x8_f16: 32x32 output tiles, K=8 per MFMA.
// A/B inputs: 2 VGPRs each (same as 16x16), C output: 16 VGPRs (was 4).
// LDS tile = 32x32x2 = 2048 bytes (4 contiguous 32x8 sub-tiles).
//
// Two-level parallelism:
//   Workgroup grid: M_WG x N_WG workgroups, num_blocks = M_WG * N_WG
//   Wave grid:      M_WAVES x N_WAVES waves per WG
//   M_DIM = M_WG * M_TILES_WG * 32  (tiles_wg decoupled from waves)
//   N_DIM = N_WG * N_TILES_WG * 32
//   M_T = M_TILES_WG / M_WAVES  (per-wave, derived)
//   N_T = N_TILES_WG / N_WAVES
//
// Each WG has M_WAVES * N_WAVES waves (threads = that * 64).
// wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
//
// LDS layout per K-iteration (single large allocations, K_T slices):
//   A: A_LDS_BYTES = M_TILES_WG * K_T * 2048 bytes
//   B: B_LDS_BYTES = N_TILES_WG * K_T * 2048 bytes
//   Per-wave offsets: base + (kt * tiles_per_slice + wave_base + i) * 2048.
//   Total: (A_LDS_BYTES + B_LDS_BYTES) per pipeline stage.
//
// Scalar substitutions:
//   M_T, N_T, K_T            - per-wave number of tiles (derived: M_TILES_WG/M_WAVES)
//   M_TILES_WG, N_TILES_WG   - per-workgroup tile counts (decoupled from waves)
//   K_TILES                  - total K-tiles (K/32), K-loop steps by K_T
//   A_TILES_PER_SLICE, B_TILES_PER_SLICE - tiles per kt-slice in LDS (= M_TILES_WG, N_TILES_WG)
//   M_WG, N_WG               - workgroup grid dimensions
//   M_WAVES, N_WAVES         - wave grid dimensions within each WG
//   A_LDS_BYTES, B_LDS_BYTES - LDS allocation sizes
//   STRIDE_AB, STRIDE_C, SHARED_MEM
//   STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE

// Register type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx16 = !amdgcn.vgpr<[? + 16]>
!rt_C_f32 = !vx16
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Memref buffer type aliases (used in helper function signatures)
!gfut_a_buf = memref<?x!future_global_read>
!gfut_b_buf = memref<?x!future_global_read>
!tok_a_buf = memref<?x!lds_write_token>
!tok_b_buf = memref<?x!lds_write_token>
!fut_a_buf = memref<?x!future_lds_read>
!fut_b_buf = memref<?x!future_lds_read>
!c_buf = memref<?x!rt_C_f32>

amdgcn.module @kittens_gemm_f16_32x32_weak_scaled target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library functions (external, provided by preload library)
  func.func private @wave_id() -> index
  func.func private @zero_C_32x32() -> !rt_C_f32
  func.func private @store_C_32x32_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // 32x32 composite primitives (from lds_32x32_f16.mlir, dwordx4 loads)
  func.func private @load_global_tile_32x32_f16(!sx2, index, index, index)
      -> (!future_global_read, !future_global_read)
  func.func private @store_global_tile_to_lds_32x32_f16(index,
      !future_global_read, !future_global_read)
      -> (!lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token)
  func.func private @wait_lds_writes_32x32(
      !lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token)
  func.func private @load_lds_A_32x32_f16(index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read)
  func.func private @load_lds_B_32x32_f16(index)
      -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read)
  func.func private @compute_mfmas_32x32(
      !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read,
      !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read,
      !rt_C_f32) -> !rt_C_f32

{{K_LOOP_HELPERS}}

  // Multi-WG multi-wave GEMM with pipelined LDS (32x32x8 MFMA)
  // M_WAVES * N_WAVES waves per WG; block_dim = (M_WAVES * N_WAVES * 64, 1, 1).
  // num_blocks = M_WG * N_WG; flat block ID delinearized into (m_wg, n_wg).
  // wave_id delinearized into (wave_m, wave_n) via (M_WAVES, N_WAVES).
  amdgcn.kernel @gemm_f16_32x32_weak_scaled arguments <[
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
    %c_M_TILES_WG = arith.constant {{M_TILES_WG}} : index
    %c_N_TILES_WG = arith.constant {{N_TILES_WG}} : index
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

    // WG owns M_TILES_WG tiles; wave_m maps to M_T consecutive tiles within the WG.
    %m_base = affine.apply affine_map<(mwg, wm)[mt_wg, mt] -> (mwg * mt_wg + wm * mt)>
        (%m_wg, %wave_m)[%c_M_TILES_WG, %c_M_T]
    %n_base = affine.apply affine_map<(nwg, wn)[nt_wg, nt] -> (nwg * nt_wg + wn * nt)>
        (%n_wg, %wave_n)[%c_N_TILES_WG, %c_N_T]
    %wave_a_base = affine.apply affine_map<(wm)[mt] -> (wm * mt)>(%wave_m)[%c_M_T]
    %wave_b_base = affine.apply affine_map<(wn)[nt] -> (wn * nt)>(%wave_n)[%c_N_T]

    // === Initialize accumulators (constexpr over M_T*N_T) ===
    %mn = affine.apply affine_map<()[m, n] -> (m * n)>()[%c_M_T, %c_N_T]
    %C_buf = memref.alloca(%mn) : !c_buf
    scf.for %i = %c0 to %mn step %c1 {
      %z = func.call @zero_C_32x32() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    scf.for %k = %c0 to %K_tiles step %c_K_T {

      // Stage GLOBAL_LOAD: allocate LDS.
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

      // Stage COMPUTE: MFMAs (constexpr over M_T x K_T x N_T)
      func.call @k_compute_mfmas(%c_M_T, %c_N_T, %c_K_T, %a_fut, %b_fut, %C_buf)
          : (index, index, index, !fut_a_buf, !fut_b_buf, !c_buf) -> ()

      // Deallocate LDS at COMPUTE stage.
      amdgcn.dealloc_lds %lds_a_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b_h {sched.stage = {{STAGE_COMPUTE}} : i32}
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_ptr, %stride_C, %m_base, %n_base)
        : (index, index, !c_buf, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
