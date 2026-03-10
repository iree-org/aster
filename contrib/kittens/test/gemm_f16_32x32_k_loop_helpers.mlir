  // === K-loop helper functions for 32x32 transfer tiles (32x32x8 MFMA) ===
  // When k_t > 1, each pipeline stage processes multiple 32x32 K-tiles.
  //
  // Each 32x32 tile = 4 contiguous 32x8 sub-tiles (2048 bytes in LDS).
  // Buffers store 4 sub-components per tile: size = k_t * dim_t * 4.
  //
  // Library functions use memref<?x...> buffers (4 entries per tile),
  // matching the 16x16 composability style from gemm_f16_k_loop_helpers.mlir.
  //
  // Uses 32x32 composite library functions:
  //   load_global_tile_32x32_f16, store_global_tile_to_lds_32x32_f16,
  //   wait_lds_writes_32x32, load_lds_A/B_32x32_f16, compute_mfmas_32x32

  // Issue global loads for A tiles across k_t 32x32 K-tiles.
  // Each load returns memref<?x!future_global_read> (4 entries per tile).
  // Flat buffer: k_t * m_t * 4 entries.
  func.func private @k_load_a_from_global(%m_t: index, %k_t: index,
      %A_ptr: !sx2, %k: index, %stride_AB: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%m_base]
        %tile_buf = func.call @load_global_tile_32x32_f16(%A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> !gfut_a_buf
        // Copy 4 sub-tile futures into flat buffer
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_a_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %gfut_a[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_a_buf
        } {aster.constexpr}
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
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%n_base]
        %tile_buf = func.call @load_global_tile_32x32_f16(%B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> !gfut_b_buf
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_b_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %gfut_b[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Store A global load futures to LDS as 32x32 tiles (2048 bytes each).
  func.func private @k_store_a_to_lds(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index,
      %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 2048)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        // Extract 4 sub-tile futures into temp buffer for library call
        %tmp_gf = memref.alloca(%c4) : !gfut_a_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %val = memref.load %gfut_a[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !gfut_a_buf
          memref.store %val, %tmp_gf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !gfut_a_buf
        } {aster.constexpr}
        %tok_buf = func.call @store_global_tile_to_lds_32x32_f16(%off, %tmp_gf)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !gfut_a_buf) -> !tok_a_buf
        // Copy 4 write tokens into flat buffer
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_buf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_a_buf
          memref.store %t, %tok_a[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_a_buf
        } {aster.constexpr}
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
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 2048)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tmp_gf = memref.alloca(%c4) : !gfut_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %val = memref.load %gfut_b[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !gfut_b_buf
          memref.store %val, %tmp_gf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !gfut_b_buf
        } {aster.constexpr}
        %tok_buf = func.call @store_global_tile_to_lds_32x32_f16(%off, %tmp_gf)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !gfut_b_buf) -> !tok_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_buf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_b_buf
          memref.store %t, %tok_b[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Wait for all A LDS write tokens (k_t * m_t * 4 tokens).
  func.func private @k_wait_lds_writes_a(%m_t: index, %k_t: index,
      %tok_a: !tok_a_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        // Extract 4 tokens into temp buffer for library call
        %tmp = memref.alloca(%c4) : !tok_a_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_a[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !tok_a_buf
          memref.store %t, %tmp[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !tok_a_buf
        } {aster.constexpr}
        func.call @wait_lds_writes_32x32(%tmp)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (!tok_a_buf) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Wait for all B LDS write tokens (k_t * n_t * 4 tokens).
  func.func private @k_wait_lds_writes_b(%n_t: index, %k_t: index,
      %tok_b: !tok_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tmp = memref.alloca(%c4) : !tok_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_b[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !tok_b_buf
          memref.store %t, %tmp[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !tok_b_buf
        } {aster.constexpr}
        func.call @wait_lds_writes_32x32(%tmp)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (!tok_b_buf) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Read A tiles from LDS (k_t * m_t 32x32 tiles, 4 futures per tile).
  func.func private @k_read_lds_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 2048)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %tile_buf = func.call @load_lds_A_32x32_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> !fut_a_buf
        // Copy 4 futures into flat buffer
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_a_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %a_fut[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS (k_t * n_t 32x32 tiles, 4 futures per tile).
  func.func private @k_read_lds_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 2048)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %tile_buf = func.call @load_lds_B_32x32_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> !fut_b_buf
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_b_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %b_fut[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Compute MFMAs: iterate over (m, k, n) tile combinations.
  // Each (m, k, n) does 4 MFMAs via compute_mfmas_32x32.
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index, %k_t: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_t, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %kt, %nt = affine.delinearize_index %idx into (%m_t, %k_t, %n_t) : index, index, index
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %a_base = affine.linearize_index [%kt, %mt] by (%k_t, %m_t) : index
      %b_base = affine.linearize_index [%kt, %nt] by (%k_t, %n_t) : index

      // Extract 4 A and 4 B futures into temp buffers for library call
      %tmp_a = memref.alloca(%c4) : !fut_a_buf
      %tmp_b = memref.alloca(%c4) : !fut_b_buf
      scf.for %s = %c0 to %c4 step %c1 {
        %a_idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%a_base, %s)
        %b_idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%b_base, %s)
        %af = memref.load %a_fut[%a_idx] {sched.stage = {{STAGE_COMPUTE}} : i32} : !fut_a_buf
        %bf = memref.load %b_fut[%b_idx] {sched.stage = {{STAGE_COMPUTE}} : i32} : !fut_b_buf
        memref.store %af, %tmp_a[%s] {sched.stage = {{STAGE_COMPUTE}} : i32} : !fut_a_buf
        memref.store %bf, %tmp_b[%s] {sched.stage = {{STAGE_COMPUTE}} : i32} : !fut_b_buf
      } {aster.constexpr}

      %c_old = memref.load %c_buf[%c_idx] {sched.stage = {{STAGE_COMPUTE}} : i32} : !c_buf
      %c_new = func.call @compute_mfmas_32x32(%tmp_a, %tmp_b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!fut_a_buf, !fut_b_buf, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] {sched.stage = {{STAGE_COMPUTE}} : i32} : !c_buf
    } {aster.constexpr}
    return
  }

  // Store C accumulator tiles to global memory.
  // store_C_32x32_f32 returns memref<?x!write_token> (16 tokens per tile).
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
        // Store tile; no explicit wait needed -- s_endpgm drains all outstanding stores.
        func.call @store_C_32x32_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !wtok_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }
