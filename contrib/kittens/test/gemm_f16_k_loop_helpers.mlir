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
  // m_base/n_base: WG's starting tile indices.
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
