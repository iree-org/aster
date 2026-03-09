  // === K-loop helper functions for 32x32 transfer tiles (32x32x8 MFMA) ===
  //
  // Split into Phase 1 (VALU: address computation) and Phase 2 (MEM: memory ops)
  // to enable kernel-level FU type batching across A and B tiles.
  //
  // Each 32x32 tile = 4 contiguous 32x8 sub-tiles (2048 bytes in LDS).
  // Flat buffers store 4 sub-components per tile: size = k_t * dim_t * 4.
  //
  // Uses split library functions:
  //   compute_global_load_addrs_32x32_f16 / issue_global_loads_32x32_f16
  //   prepare_lds_write_32x32_f16 / issue_lds_writes_32x32_f16
  //   compute_lds_A/B_addrs_32x32_f16 / issue_lds_reads_A/B_32x32_f16
  //   wait_lds_writes_32x32, compute_mfmas_32x32

  //===--------------------------------------------------------------------===//
  // Global Load Phase 1: Compute addresses (VALU)
  //===--------------------------------------------------------------------===//

  // Compute global load addresses for A tiles.
  // Returns flat addr_buf and dst_buf (k_t * m_t * 4 entries each).
  func.func private @k_compute_global_addrs_a(%m_t: index, %k_t: index,
      %A_ptr: !sx2, %k: index, %stride_AB: index, %m_base: index)
      -> (memref<?x!vx2>, memref<?x!vx2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!vx2>
    %flat_dsts = memref.alloca(%buf_size) : memref<?x!vx2>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%m_base]
        %tile_addrs, %tile_dsts = func.call @compute_global_load_addrs_32x32_f16(
            %A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> (memref<?x!vx2>, memref<?x!vx2>)
        // Copy 4 entries into flat buffers
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          %d = memref.load %tile_dsts[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_dsts[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_addrs, %flat_dsts : memref<?x!vx2>, memref<?x!vx2>
  }

  // Compute global load addresses for B tiles.
  func.func private @k_compute_global_addrs_b(%n_t: index, %k_t: index,
      %B_ptr: !sx2, %k: index, %stride_AB: index, %n_base: index)
      -> (memref<?x!vx2>, memref<?x!vx2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!vx2>
    %flat_dsts = memref.alloca(%buf_size) : memref<?x!vx2>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%i)[%n_base]
        %tile_addrs, %tile_dsts = func.call @compute_global_load_addrs_32x32_f16(
            %B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (!sx2, index, index, index) -> (memref<?x!vx2>, memref<?x!vx2>)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          %d = memref.load %tile_dsts[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_dsts[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_addrs, %flat_dsts : memref<?x!vx2>, memref<?x!vx2>
  }

  //===--------------------------------------------------------------------===//
  // Global Load Phase 2: Issue loads (VMEM)
  //===--------------------------------------------------------------------===//

  // Issue global loads for A tiles from pre-computed addresses.
  func.func private @k_issue_global_loads_a(%m_t: index, %k_t: index,
      %flat_addrs: memref<?x!vx2>, %flat_dsts: memref<?x!vx2>)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        // Extract 4 entries from flat buffers into temp buffers
        %tmp_addrs = memref.alloca(%c4) : memref<?x!vx2>
        %tmp_dsts = memref.alloca(%c4) : memref<?x!vx2>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          %d = memref.load %flat_dsts[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_dsts[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
        %tile_buf = func.call @issue_global_loads_32x32_f16(%tmp_addrs, %tmp_dsts)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (memref<?x!vx2>, memref<?x!vx2>) -> !gfut_a_buf
        // Copy 4 futures into flat buffer
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_a_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %gfut_a[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads for B tiles from pre-computed addresses.
  func.func private @k_issue_global_loads_b(%n_t: index, %k_t: index,
      %flat_addrs: memref<?x!vx2>, %flat_dsts: memref<?x!vx2>)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tmp_addrs = memref.alloca(%c4) : memref<?x!vx2>
        %tmp_dsts = memref.alloca(%c4) : memref<?x!vx2>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          %d = memref.load %flat_dsts[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_dsts[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
        %tile_buf = func.call @issue_global_loads_32x32_f16(%tmp_addrs, %tmp_dsts)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (memref<?x!vx2>, memref<?x!vx2>) -> !gfut_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_b_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %gfut_b[%idx] {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32} : !gfut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // LDS Write Phase 1: Prepare data + compute addresses (VALU)
  //===--------------------------------------------------------------------===//

  // Prepare LDS writes for A tiles: extract global data + compute LDS addresses.
  // Returns flat data_buf and addr_buf (k_t * m_t * 4 entries each).
  func.func private @k_prepare_lds_writes_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index,
      %gfut_a: !gfut_a_buf) -> (memref<?x!vx2>, memref<?x!v>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %flat_data = memref.alloca(%buf_size) : memref<?x!vx2>
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!v>
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
        %tile_data, %tile_addrs = func.call @prepare_lds_write_32x32_f16(%off, %tmp_gf)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !gfut_a_buf) -> (memref<?x!vx2>, memref<?x!v>)
        // Copy 4 entries into flat buffers
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %d = memref.load %tile_data[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_data[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_data, %flat_addrs : memref<?x!vx2>, memref<?x!v>
  }

  // Prepare LDS writes for B tiles.
  func.func private @k_prepare_lds_writes_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index,
      %gfut_b: !gfut_b_buf) -> (memref<?x!vx2>, memref<?x!v>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %flat_data = memref.alloca(%buf_size) : memref<?x!vx2>
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!v>
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
        %tile_data, %tile_addrs = func.call @prepare_lds_write_32x32_f16(%off, %tmp_gf)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index, !gfut_b_buf) -> (memref<?x!vx2>, memref<?x!v>)
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %d = memref.load %tile_data[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_data[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_data, %flat_addrs : memref<?x!vx2>, memref<?x!v>
  }

  //===--------------------------------------------------------------------===//
  // LDS Write Phase 2: Issue writes (DS)
  //===--------------------------------------------------------------------===//

  // Issue LDS writes for A tiles from pre-computed data and addresses.
  func.func private @k_issue_lds_writes_a(%m_t: index, %k_t: index,
      %flat_data: memref<?x!vx2>, %flat_addrs: memref<?x!v>)
      -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %tmp_data = memref.alloca(%c4) : memref<?x!vx2>
        %tmp_addrs = memref.alloca(%c4) : memref<?x!v>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %d = memref.load %flat_data[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_data[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
        } {aster.constexpr}
        %tok_buf = func.call @issue_lds_writes_32x32_f16(%tmp_data, %tmp_addrs)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (memref<?x!vx2>, memref<?x!v>) -> !tok_a_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_buf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_a_buf
          memref.store %t, %tok_a[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Issue LDS writes for B tiles from pre-computed data and addresses.
  func.func private @k_issue_lds_writes_b(%n_t: index, %k_t: index,
      %flat_data: memref<?x!vx2>, %flat_addrs: memref<?x!v>)
      -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tmp_data = memref.alloca(%c4) : memref<?x!vx2>
        %tmp_addrs = memref.alloca(%c4) : memref<?x!v>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %d = memref.load %flat_data[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_data[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!vx2>
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : memref<?x!v>
        } {aster.constexpr}
        %tok_buf = func.call @issue_lds_writes_32x32_f16(%tmp_data, %tmp_addrs)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (memref<?x!vx2>, memref<?x!v>) -> !tok_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %t = memref.load %tok_buf[%s] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_b_buf
          memref.store %t, %tok_b[%idx] {sched.stage = {{STAGE_DS_WRITE}} : i32} : !tok_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  //===--------------------------------------------------------------------===//
  // LDS Write Wait (unchanged from original)
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  // LDS Read Phase 1: Compute addresses (VALU)
  //===--------------------------------------------------------------------===//

  // Compute LDS read addresses for A tiles.
  // Returns flat addr_buf and dst_buf (k_t * m_t * 4 entries each).
  func.func private @k_compute_lds_read_addrs_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index)
      -> (memref<?x!v>, memref<?x!vx2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!v>
    %flat_dsts = memref.alloca(%buf_size) : memref<?x!vx2>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 2048)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %tile_addrs, %tile_dsts = func.call @compute_lds_A_addrs_32x32_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> (memref<?x!v>, memref<?x!vx2>)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          %d = memref.load %tile_dsts[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_dsts[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_addrs, %flat_dsts : memref<?x!v>, memref<?x!vx2>
  }

  // Compute LDS read addresses for B tiles.
  func.func private @k_compute_lds_read_addrs_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index)
      -> (memref<?x!v>, memref<?x!vx2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %flat_addrs = memref.alloca(%buf_size) : memref<?x!v>
    %flat_dsts = memref.alloca(%buf_size) : memref<?x!vx2>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 2048)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %tile_addrs, %tile_dsts = func.call @compute_lds_B_addrs_32x32_f16(%off)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (index) -> (memref<?x!v>, memref<?x!vx2>)
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %tile_addrs[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          memref.store %a, %flat_addrs[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          %d = memref.load %tile_dsts[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
          memref.store %d, %flat_dsts[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %flat_addrs, %flat_dsts : memref<?x!v>, memref<?x!vx2>
  }

  //===--------------------------------------------------------------------===//
  // LDS Read Phase 2: Issue reads (DS)
  //===--------------------------------------------------------------------===//

  // Issue LDS reads for A tiles from pre-computed addresses.
  func.func private @k_issue_lds_reads_a(%m_t: index, %k_t: index,
      %flat_addrs: memref<?x!v>, %flat_dsts: memref<?x!vx2>)
      -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %tmp_addrs = memref.alloca(%c4) : memref<?x!v>
        %tmp_dsts = memref.alloca(%c4) : memref<?x!vx2>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          %d = memref.load %flat_dsts[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_dsts[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
        %tile_buf = func.call @issue_lds_reads_A_32x32_f16(%tmp_addrs, %tmp_dsts)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (memref<?x!v>, memref<?x!vx2>) -> !fut_a_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_a_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %a_fut[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Issue LDS reads for B tiles from pre-computed addresses.
  func.func private @k_issue_lds_reads_b(%n_t: index, %k_t: index,
      %flat_addrs: memref<?x!v>, %flat_dsts: memref<?x!vx2>)
      -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %base = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %tmp_addrs = memref.alloca(%c4) : memref<?x!v>
        %tmp_dsts = memref.alloca(%c4) : memref<?x!vx2>
        scf.for %s = %c0 to %c4 step %c1 {
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          %a = memref.load %flat_addrs[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          memref.store %a, %tmp_addrs[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!v>
          %d = memref.load %flat_dsts[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
          memref.store %d, %tmp_dsts[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : memref<?x!vx2>
        } {aster.constexpr}
        %tile_buf = func.call @issue_lds_reads_B_32x32_f16(%tmp_addrs, %tmp_dsts)
            {sched.stage = {{STAGE_DS_READ}} : i32}
            : (memref<?x!v>, memref<?x!vx2>) -> !fut_b_buf
        scf.for %s = %c0 to %c4 step %c1 {
          %val = memref.load %tile_buf[%s] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_b_buf
          %idx = affine.apply affine_map<(b, s) -> (b * 4 + s)>(%base, %s)
          memref.store %val, %b_fut[%idx] {sched.stage = {{STAGE_DS_READ}} : i32} : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // Compute MFMAs (unchanged from original)
  //===--------------------------------------------------------------------===//

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

  //===--------------------------------------------------------------------===//
  // Store C tiles (unchanged from original)
  //===--------------------------------------------------------------------===//

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
        %tok_buf = func.call @store_C_32x32_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !wtok_buf
        func.call @wait_global_writes_32x32(%tok_buf) : (!wtok_buf) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }
