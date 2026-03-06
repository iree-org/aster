  // === K-loop helper functions for 32x32 transfer tiles (16x16x16 MFMA) ===
  // When k_t > 1, each pipeline stage processes multiple 32x32 K-tiles.
  // Each 32x32 tile = 2048 bytes in LDS.
  // Global loads use dwordx4: 2 futures per tile.
  // LDS writes: 4 tokens per tile (split each dwordx4 into 2 x ds_write_b64).
  //
  // Uses 32x32 composite library functions for transfer tiles, and _m16
  // variants for LDS read / compute / store (16x16x16 MFMA from 32x32 LDS):
  //   load_global_tile_32x32_f16, store_global_tile_to_lds_32x32_f16,
  //   wait_lds_writes_32x32, load_lds_A/B_32x32_m16_f16,
  //   compute_mfmas_32x32_m16, store_C_32x32_m16_f32

  // Issue global loads for A tiles across k_t 32x32 K-tiles.
  // Each load returns 2 dwordx4 futures stored at consecutive buffer indices.
  func.func private @k_load_a_from_global(%m_t: index, %k_t: index,
      %A_ptr: !sx2, %k: index, %stride_AB: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%m_base]
        %f0, %f1 = func.call @load_global_tile_32x32_f16(%A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index)
            -> (!future_global_read, !future_global_read)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 2)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 2 + 1)>(%base)
        memref.store %f0, %gfut_a[%idx0] : !gfut_a_buf
        memref.store %f1, %gfut_a[%idx1] : !gfut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads for B tiles across k_t 32x32 K-tiles.
  func.func private @k_load_b_from_global(%n_t: index, %k_t: index,
      %B_ptr: !sx2, %k: index, %stride_AB: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%n_base]
        %f0, %f1 = func.call @load_global_tile_32x32_f16(%B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index)
            -> (!future_global_read, !future_global_read)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 2)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 2 + 1)>(%base)
        memref.store %f0, %gfut_b[%idx0] : !gfut_b_buf
        memref.store %f1, %gfut_b[%idx1] : !gfut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Store A global load futures to LDS as 32x32 tiles (2048 bytes each).
  // Each tile has 2 global futures -> 4 LDS write tokens.
  func.func private @k_store_a_to_lds(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index,
      %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%tok_buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 2048)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %gbase = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gidx0 = affine.apply affine_map<(b) -> (b * 2)>(%gbase)
        %gidx1 = affine.apply affine_map<(b) -> (b * 2 + 1)>(%gbase)
        %gf0 = memref.load %gfut_a[%gidx0] : !gfut_a_buf
        %gf1 = memref.load %gfut_a[%gidx1] : !gfut_a_buf
        %t0, %t1, %t2, %t3 = func.call @store_global_tile_to_lds_32x32_f16(%off, %gf0, %gf1)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !future_global_read, !future_global_read)
            -> (!lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token)
        %tidx0 = affine.apply affine_map<(b) -> (b * 4)>(%gbase)
        %tidx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%gbase)
        %tidx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%gbase)
        %tidx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%gbase)
        memref.store %t0, %tok_a[%tidx0] : !tok_a_buf
        memref.store %t1, %tok_a[%tidx1] : !tok_a_buf
        memref.store %t2, %tok_a[%tidx2] : !tok_a_buf
        memref.store %t3, %tok_a[%tidx3] : !tok_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Store B global load futures to LDS as 32x32 tiles (2048 bytes each).
  func.func private @k_store_b_to_lds(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index,
      %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%tok_buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 2048)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %gbase = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %gidx0 = affine.apply affine_map<(b) -> (b * 2)>(%gbase)
        %gidx1 = affine.apply affine_map<(b) -> (b * 2 + 1)>(%gbase)
        %gf0 = memref.load %gfut_b[%gidx0] : !gfut_b_buf
        %gf1 = memref.load %gfut_b[%gidx1] : !gfut_b_buf
        %t0, %t1, %t2, %t3 = func.call @store_global_tile_to_lds_32x32_f16(%off, %gf0, %gf1)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !future_global_read, !future_global_read)
            -> (!lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token)
        %tidx0 = affine.apply affine_map<(b) -> (b * 4)>(%gbase)
        %tidx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%gbase)
        %tidx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%gbase)
        %tidx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%gbase)
        memref.store %t0, %tok_b[%tidx0] : !tok_b_buf
        memref.store %t1, %tok_b[%tidx1] : !tok_b_buf
        memref.store %t2, %tok_b[%tidx2] : !tok_b_buf
        memref.store %t3, %tok_b[%tidx3] : !tok_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Wait for all A LDS write tokens (k_t * m_t * 4 tokens).
  func.func private @k_wait_lds_writes_a(%m_t: index, %k_t: index,
      %tok_a: !tok_a_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 4)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%base)
        %idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%base)
        %idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%base)
        %t0 = memref.load %tok_a[%idx0] : !tok_a_buf
        %t1 = memref.load %tok_a[%idx1] : !tok_a_buf
        %t2 = memref.load %tok_a[%idx2] : !tok_a_buf
        %t3 = memref.load %tok_a[%idx3] : !tok_a_buf
        func.call @wait_lds_writes_32x32(%t0, %t1, %t2, %t3)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (!lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Wait for all B LDS write tokens (k_t * n_t * 4 tokens).
  func.func private @k_wait_lds_writes_b(%n_t: index, %k_t: index,
      %tok_b: !tok_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 4)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%base)
        %idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%base)
        %idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%base)
        %t0 = memref.load %tok_b[%idx0] : !tok_b_buf
        %t1 = memref.load %tok_b[%idx1] : !tok_b_buf
        %t2 = memref.load %tok_b[%idx2] : !tok_b_buf
        %t3 = memref.load %tok_b[%idx3] : !tok_b_buf
        func.call @wait_lds_writes_32x32(%t0, %t1, %t2, %t3)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (!lds_write_token, !lds_write_token, !lds_write_token, !lds_write_token) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Read A tiles from LDS (k_t * m_t 32x32 tiles, 4 futures per tile).
  func.func private @k_read_lds_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 2048)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %f0, %f1, %f2, %f3 = func.call @load_lds_A_32x32_m16_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 4)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%base)
        %idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%base)
        %idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%base)
        memref.store %f0, %a_fut[%idx0] : !fut_a_buf
        memref.store %f1, %a_fut[%idx1] : !fut_a_buf
        memref.store %f2, %a_fut[%idx2] : !fut_a_buf
        memref.store %f3, %a_fut[%idx3] : !fut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS (k_t * n_t 32x32 tiles, 4 futures per tile).
  func.func private @k_read_lds_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 2048)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %f0, %f1, %f2, %f3 = func.call @load_lds_B_32x32_m16_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %idx0 = affine.apply affine_map<(b) -> (b * 4)>(%base)
        %idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%base)
        %idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%base)
        %idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%base)
        memref.store %f0, %b_fut[%idx0] : !fut_b_buf
        memref.store %f1, %b_fut[%idx1] : !fut_b_buf
        memref.store %f2, %b_fut[%idx2] : !fut_b_buf
        memref.store %f3, %b_fut[%idx3] : !fut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Compute MFMAs: iterate over (m, k, n) tile combinations.
  // Each (m, k, n) does 8 MFMAs via compute_mfmas_32x32_m16.
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index, %k_t: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_t, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %kt, %nt = affine.delinearize_index %idx into (%m_t, %k_t, %n_t) : index, index, index
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %a_base = affine.linearize_index [%kt, %mt] by (%k_t, %m_t) : index
      %b_base = affine.linearize_index [%kt, %nt] by (%k_t, %n_t) : index

      // Load 4 A and 4 B sub-tile futures for this 32x32 tile combination
      %a_idx0 = affine.apply affine_map<(b) -> (b * 4)>(%a_base)
      %a_idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%a_base)
      %a_idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%a_base)
      %a_idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%a_base)
      %af0 = memref.load %a_fut[%a_idx0] : !fut_a_buf
      %af1 = memref.load %a_fut[%a_idx1] : !fut_a_buf
      %af2 = memref.load %a_fut[%a_idx2] : !fut_a_buf
      %af3 = memref.load %a_fut[%a_idx3] : !fut_a_buf

      %b_idx0 = affine.apply affine_map<(b) -> (b * 4)>(%b_base)
      %b_idx1 = affine.apply affine_map<(b) -> (b * 4 + 1)>(%b_base)
      %b_idx2 = affine.apply affine_map<(b) -> (b * 4 + 2)>(%b_base)
      %b_idx3 = affine.apply affine_map<(b) -> (b * 4 + 3)>(%b_base)
      %bf0 = memref.load %b_fut[%b_idx0] : !fut_b_buf
      %bf1 = memref.load %b_fut[%b_idx1] : !fut_b_buf
      %bf2 = memref.load %b_fut[%b_idx2] : !fut_b_buf
      %bf3 = memref.load %b_fut[%b_idx3] : !fut_b_buf

      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @compute_mfmas_32x32_m16(%af0, %af1, %af2, %af3, %bf0, %bf1, %bf2, %bf3, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read,
             !future_lds_read, !future_lds_read, !future_lds_read, !future_lds_read,
             !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }

  // Store C accumulator tiles to global memory.
  func.func private @store_c_tiles(%m_t: index, %n_t: index,
      %c_buf: !c_buf, %C_ptr: !sx2, %stride_C: index,
      %m_base: index, %n_base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
        %c_tile = memref.load %c_buf[%idx] : !c_buf
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%mt)[%m_base]
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%nt)[%n_base]
        %tok = func.call @store_C_32x32_m16_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !write_token
        amdgcn.wait deps %tok : !write_token
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }
