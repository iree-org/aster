  // From lds_mfma_16x64_b.mlir
  func.func private @load_lds_A_swizzled(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled(index, index, index) -> !future_lds_read
  // Decoupled address computation functions
  func.func private @compute_global_byte_off_16x64_b(index, index, index) -> index
  func.func private @load_global_at_byte_off(!aster_utils.any, index) -> !future_global_read
  func.func private @compute_lds_write_addrs_16x64_b(index) -> (index, index)
  func.func private @get_global_load_value_vx4(!future_global_read) -> !vx4
  func.func private @write_vx4_to_lds_at(!vx4, index, index) -> (!lds_write_token, !lds_write_token)
  func.func private @compute_lds_read_addr_A(index, index, index) -> index
  func.func private @compute_lds_read_addr_B(index, index, index) -> index
  func.func private @read_vx2_from_lds_at(index) -> !future_lds_read

  // === K-loop helper functions for 16x32 transfer tiles (dwordx4 global loads) ===
  // Each 16x32 transfer tile covers K=32 and yields 2 MFMA K-steps of 16 each.
  // Global loads use dwordx4 (16 bytes/lane).
  // LDS writes: 2x ds_write_b64 per tile with XOR swizzle (split vx4 -> 2x vx2).
  // LDS reads use ds_read_b64 for each K-half (K0/K1 as separate 16x16 XOR tiles).
  //
  // Buffer sizes:
  //   Global load futures:     k_t * dim_t (one per 16x32 tile)
  //      LDS write tokens: 2 * k_t * dim_t (two per tile: lo/hi halves)
  //      LDS read futures: 2 * k_t * dim_t (two per tile: K0, K1)
  //       Register values: 2 * k_t * dim_t (two per tile: K0, K1)
  //
  // LDS tile size: 1024 bytes (16x64_b layout: 16 rows x 64 bytes)
  // K0 at byte-offset 0, K1 at byte-offset 32.

  //===--------------------------------------------------------------------===//
  // 1. Global load + decoupled
  //===--------------------------------------------------------------------===//

  // Issue dwordx4 global loads for A tiles across k_t K-tiles.
  // Each tile covers a 16x64_b = 32 K-elements (for f16).
  func.func private @k_load_a_16x32_from_global(%m_t: index, %k_t: index,
      %A_ptr: !aster_utils.any, %k: index, %stride_AB: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%m_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %fut = func.call @load_global_tile_16x64_b(%A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{A_STAGE_LOAD}} : i32}
            : (!aster_utils.any, index, index, index) -> !future_global_read
        memref.store %fut, %gfut_a[%idx] : !gfut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue dwordx4 global loads for B tiles across k_t K-tiles.
  // Each tile covers a 16x64_b = 32 K-elements (for f16).
  func.func private @k_load_b_16x32_from_global(%n_t: index, %k_t: index,
      %B_ptr: !aster_utils.any, %k: index, %stride_AB: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%n_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %fut = func.call @load_global_tile_16x64_b(%B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{A_STAGE_LOAD}} : i32}
            : (!aster_utils.any, index, index, index) -> !future_global_read
        memref.store %fut, %gfut_b[%idx] : !gfut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Compute global load byte offsets for A tiles (no loads issued).
  func.func private @k_compute_global_addrs_a(%m_t: index, %k_t: index,
      %k: index, %stride_AB: index, %m_base: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t step %c1 {
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%m_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %byte_off = func.call @compute_global_byte_off_16x64_b(%m_off, %k_offset, %stride_AB)
            {sched.stage = {{A_STAGE_LOAD}} : i32}
            : (index, index, index) -> index
        memref.store %byte_off, %addrs[%idx] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Compute global load byte offsets for B tiles (no loads issued).
  func.func private @k_compute_global_addrs_b(%n_t: index, %k_t: index,
      %k: index, %stride_AB: index, %n_base: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t step %c1 {
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%i)[%n_base]
        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %byte_off = func.call @compute_global_byte_off_16x64_b(%n_off, %k_offset, %stride_AB)
            {sched.stage = {{A_STAGE_LOAD}} : i32}
            : (index, index, index) -> index
        memref.store %byte_off, %addrs[%idx] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Issue global loads at pre-computed byte offsets for A tiles.
  func.func private @k_issue_global_loads_a(%addrs: memref<?xindex>,
      %A_ptr: !aster_utils.any) -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %byte_off = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @load_global_at_byte_off(%A_ptr, %byte_off)
          {sched.stage = {{A_STAGE_LOAD}} : i32}
          : (!aster_utils.any, index) -> !future_global_read
      memref.store %fut, %gfut_a[%idx] : !gfut_a_buf
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads at pre-computed byte offsets for B tiles.
  func.func private @k_issue_global_loads_b(%addrs: memref<?xindex>,
      %B_ptr: !aster_utils.any) -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %byte_off = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @load_global_at_byte_off(%B_ptr, %byte_off)
          {sched.stage = {{A_STAGE_LOAD}} : i32}
          : (!aster_utils.any, index) -> !future_global_read
      memref.store %fut, %gfut_b[%idx] : !gfut_b_buf
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 2. LDS write + decoupled
  //===--------------------------------------------------------------------===//

  // Wait for A global loads and store to LDS with 2x ds_write_b64 (XOR swizzle).
  // LDS layout: k_t slices, each with tiles_per_slice tiles.
  // Offset = base + (kt * tiles_per_slice + wave_base + i) * 1024.
  // Each store returns 2 tokens -> token buffer = 2 * k_t * m_t.
  func.func private @k_store_a_16x32_to_lds(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index,
      %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%tok_buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gfut = memref.load %gfut_a[%gfut_idx] : !gfut_a_buf
        %tok_lo, %tok_hi = func.call @store_global_tile_to_lds_16x64_b(%off, %gfut)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
        // Store both tokens at consecutive indices: 2*(kt*m_t+i) and 2*(kt*m_t+i)+1
        %tok_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%m_t]
        %tok_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%tok_idx_lo, %i)
        %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%tok_idx_lo2)
        memref.store %tok_lo, %tok_a[%tok_idx_lo2] : !tok_a_buf
        memref.store %tok_hi, %tok_a[%tok_idx_hi] : !tok_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Wait for B global loads and store to LDS with 2x ds_write_b64 (XOR swizzle).
  func.func private @k_store_b_16x32_to_lds(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index,
      %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%tok_buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %gfut = memref.load %gfut_b[%gfut_idx] : !gfut_b_buf
        %tok_lo, %tok_hi = func.call @store_global_tile_to_lds_16x64_b(%off, %gfut)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
        %tok_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%n_t]
        %tok_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%tok_idx_lo, %i)
        %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%tok_idx_lo2)
        memref.store %tok_lo, %tok_b[%tok_idx_lo2] : !tok_b_buf
        memref.store %tok_hi, %tok_b[%tok_idx_hi] : !tok_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Compute LDS write address pairs for A tiles (no writes issued).
  // Returns buffer of size 2 * k_t * m_t (lo/hi address pairs interleaved).
  func.func private @k_compute_lds_write_addrs_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %m_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        %addr_lo, %addr_hi = func.call @compute_lds_write_addrs_16x64_b(%off)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (index) -> (index, index)
        %addr_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%m_t]
        %addr_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%addr_idx_lo, %i)
        %addr_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%addr_idx_lo2)
        memref.store %addr_lo, %addrs[%addr_idx_lo2] : memref<?xindex>
        memref.store %addr_hi, %addrs[%addr_idx_hi] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Compute LDS write address pairs for B tiles (no writes issued).
  func.func private @k_compute_lds_write_addrs_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %n_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        %addr_lo, %addr_hi = func.call @compute_lds_write_addrs_16x64_b(%off)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (index) -> (index, index)
        %addr_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%n_t]
        %addr_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%addr_idx_lo, %i)
        %addr_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%addr_idx_lo2)
        memref.store %addr_lo, %addrs[%addr_idx_lo2] : memref<?xindex>
        memref.store %addr_hi, %addrs[%addr_idx_hi] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Wait for global loads and store to LDS at pre-computed addresses.
  func.func private @k_store_to_lds_at_addrs_a(%addrs: memref<?xindex>,
      %m_t: index, %k_t: index, %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %m_t]
    %tok_a = memref.alloca(%tok_buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gfut = memref.load %gfut_a[%gfut_idx] : !gfut_a_buf
        %loaded = func.call @get_global_load_value_vx4(%gfut)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %tok_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%m_t]
        %tok_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%tok_idx_lo, %i)
        %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%tok_idx_lo2)
        %addr_lo = memref.load %addrs[%tok_idx_lo2] : memref<?xindex>
        %addr_hi = memref.load %addrs[%tok_idx_hi] : memref<?xindex>
        %tok_lo, %tok_hi = func.call @write_vx4_to_lds_at(%loaded, %addr_lo, %addr_hi)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo, %tok_a[%tok_idx_lo2] : !tok_a_buf
        memref.store %tok_hi, %tok_a[%tok_idx_hi] : !tok_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Wait for global loads and store to LDS at pre-computed addresses.
  func.func private @k_store_to_lds_at_addrs_b(%addrs: memref<?xindex>,
      %n_t: index, %k_t: index, %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %n_t]
    %tok_b = memref.alloca(%tok_buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %gfut = memref.load %gfut_b[%gfut_idx] : !gfut_b_buf
        %loaded = func.call @get_global_load_value_vx4(%gfut)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %tok_idx_lo = affine.apply affine_map<(d0)[s0] -> (d0 * s0 * 2)>(%kt)[%n_t]
        %tok_idx_lo2 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%tok_idx_lo, %i)
        %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 + 1)>(%tok_idx_lo2)
        %addr_lo = memref.load %addrs[%tok_idx_lo2] : memref<?xindex>
        %addr_hi = memref.load %addrs[%tok_idx_hi] : memref<?xindex>
        %tok_lo, %tok_hi = func.call @write_vx4_to_lds_at(%loaded, %addr_lo, %addr_hi)
            {sched.stage = {{A_STAGE_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo, %tok_b[%tok_idx_lo2] : !tok_b_buf
        memref.store %tok_hi, %tok_b[%tok_idx_hi] : !tok_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 3. LDS write wait + LDS read + decoupled
  //===--------------------------------------------------------------------===//

  // Wait all LDS write tokens in buf (size determined from memref.dim).
  func.func private @k_wait_lds_writes(%tok_buf: memref<?x!lds_write_token>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_count = memref.dim %tok_buf, %c0 : memref<?x!lds_write_token>
    scf.for %idx = %c0 to %tok_count step %c1 {
      %tok = memref.load %tok_buf[%idx] : memref<?x!lds_write_token>
      amdgcn.wait deps %tok {sched.stage = {{A_STAGE_READ}} : i32} : !lds_write_token
    } {aster.constexpr}
    return
  }

  // Read A tiles from LDS: k_t * 2 * m_t futures (K0 + K1 for each 16x32 tile).
  // Buffer indexed as [k_mfma, i] where k_mfma = kt*2 + kh (kh=0 for K0, 1 for K1).
  // k_byte_offset = kh * 32: K0 at byte 0, K1 at byte 32 within each LDS row.
  func.func private @k_read_lds_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %m_t]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        // Loop over K halves: kh=0 -> k_byte_offset=0 (K0), kh=1 -> k_byte_offset=32 (K1)
        scf.for %kh = %c0 to %c2 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %m_t) : index
          %fut = func.call @load_lds_A_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{A_STAGE_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS: k_t * 2 * n_t futures (K0 + K1 for each 16x32 tile).
  func.func private @k_read_lds_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %n_t]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        scf.for %kh = %c0 to %c2 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %n_t) : index
          %fut = func.call @load_lds_B_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{A_STAGE_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Compute LDS read addresses for A tiles (no reads issued).
  func.func private @k_compute_lds_read_addrs_a(%m_t: index, %k_t: index,
      %base_a: index, %wave_a_base: index, %tiles_per_slice: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %m_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, i)[base, wab, tps] -> (base + (kt * tps + wab + i) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base, %tiles_per_slice]
        scf.for %kh = %c0 to %c2 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %m_t) : index
          %addr = func.call @compute_lds_read_addr_A(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{A_STAGE_READ}} : i32}
              : (index, index, index) -> index
          memref.store %addr, %addrs[%buf_idx] : memref<?xindex>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Compute LDS read addresses for B tiles (no reads issued).
  func.func private @k_compute_lds_read_addrs_b(%n_t: index, %k_t: index,
      %base_b: index, %wave_b_base: index, %tiles_per_slice: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %n_t]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, i)[base, wbb, tps] -> (base + (kt * tps + wbb + i) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base, %tiles_per_slice]
        scf.for %kh = %c0 to %c2 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %n_t) : index
          %addr = func.call @compute_lds_read_addr_B(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{A_STAGE_READ}} : i32}
              : (index, index, index) -> index
          memref.store %addr, %addrs[%buf_idx] : memref<?xindex>
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Issue LDS reads at pre-computed addresses for A tiles.
  func.func private @k_read_lds_at_addrs_a(%addrs: memref<?xindex>) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %addr = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @read_vx2_from_lds_at(%addr)
          {sched.stage = {{A_STAGE_READ}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %a_fut[%idx] : !fut_a_buf
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Issue LDS reads at pre-computed addresses for B tiles.
  func.func private @k_read_lds_at_addrs_b(%addrs: memref<?xindex>) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %addr = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @read_vx2_from_lds_at(%addr)
          {sched.stage = {{A_STAGE_READ}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %b_fut[%idx] : !fut_b_buf
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 4. LDS read wait + MFMA + decoupled
  //===--------------------------------------------------------------------===//

  // Fused wait + MFMA: delinearize flat index into (k, m, n), wait for
  // the corresponding A[k,m] and B[k,n] futures, then MFMA.
  // k is outermost so A[k,m] is reused across all n tiles before advancing k,
  // and consecutive MFMAs update different C accumulators to hide MFMA latency.
  // Redundant waits on already-completed tokens are free (waitcnt satisfied).
  // k_mfma is the total number of MFMA K-steps (e.g. k_t * 2 for 16x32 tiles).
  func.func private @k_wait_and_compute_mfmas(%m_t: index, %n_t: index, %k_mfma: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %km, %mt, %nt = affine.delinearize_index %idx into (%k_mfma, %m_t, %n_t) : index, index, index
      %a_idx = affine.linearize_index [%km, %mt] by (%k_mfma, %m_t) : index
      %b_idx = affine.linearize_index [%km, %nt] by (%k_mfma, %n_t) : index

      // Wait + extract A and B (redundant waits are no-ops)
      %fut_a = memref.load %a_fut[%a_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %fut_b = memref.load %b_fut[%b_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16

      // MFMA
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }

  // Extract A register values from LDS read futures (flat loop, size from memref.dim).
  func.func private @k_extract_lds_values_a(%a_fut: !fut_a_buf) -> !vals_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %a_fut, %c0 : !fut_a_buf
    %a_vals = memref.alloca(%buf_size) : !vals_a_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %fut = memref.load %a_fut[%idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      memref.store %a, %a_vals[%idx] : !vals_a_buf
    } {aster.constexpr}
    return %a_vals : !vals_a_buf
  }

  // Extract B register values from LDS read futures (flat loop, size from memref.dim).
  func.func private @k_extract_lds_values_b(%b_fut: !fut_b_buf) -> !vals_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %b_fut, %c0 : !fut_b_buf
    %b_vals = memref.alloca(%buf_size) : !vals_b_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %fut = memref.load %b_fut[%idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      memref.store %b, %b_vals[%idx] : !vals_b_buf
    } {aster.constexpr}
    return %b_vals : !vals_b_buf
  }

  // Compute MFMAs: k_mfma * m_t * n_t operations with k outermost.
  // k_mfma is the total number of MFMA K-steps; caller computes it (e.g. k_t * 2 for 16x32 tiles).
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index, %k_mfma: index,
      %a_vals: !vals_a_buf, %b_vals: !vals_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %km, %mt, %nt = affine.delinearize_index %idx into (%k_mfma, %m_t, %n_t) : index, index, index
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %a_idx = affine.linearize_index [%km, %mt] by (%k_mfma, %m_t) : index
      %b_idx = affine.linearize_index [%km, %nt] by (%k_mfma, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %a = memref.load %a_vals[%a_idx] : !vals_a_buf
      %b = memref.load %b_vals[%b_idx] : !vals_b_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{A_STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 5. Global store (epilogue)
  //===--------------------------------------------------------------------===//

  // Store C accumulator tiles to global memory.
  // m_base/n_base: WG's starting tile indices.
  func.func private @store_c_tiles(%m_t: index, %n_t: index,
      %c_buf: !c_buf, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_base: index, %n_base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
        %c_tile = memref.load %c_buf[%idx] : !c_buf
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%mt)[%m_base]
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 16)>(%nt)[%n_base]
        // Store tile; no explicit wait needed -- s_endpgm drains all outstanding stores.
        func.call @store_global_C_mfma_f32_16x16x16_f16(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !aster_utils.any, index, index, index) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 6. Flat wave distribution helpers
  //===--------------------------------------------------------------------===//
  // Wave wid owns flat A-tile indices [flat_a_start, flat_a_start + a_per_wave)
  // over the (k_t * m_t_ld) A-tile space, where m_t_ld = m_tile // 16.
  // Flat index f maps to kt = f / m_t_ld, m_slot = f % m_t_ld.
  // LDS layout: (kt * m_t_ld + m_slot) * 1024 + base -- identical to old layout.

  // Issue dwordx4 global loads for a flat range of A tiles.
  func.func private @k_load_a_flat(%a_per_wave: index, %k_t: index, %m_t_ld: index,
      %A_ptr: !aster_utils.any, %k: index, %stride_AB: index,
      %m_global_base: index, %flat_a_start: index) -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %gfut_a = memref.alloca(%a_per_wave) : !gfut_a_buf
    scf.for %i = %c0 to %a_per_wave step %c1 {
      %flat_a = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_a_start]
      %kt = arith.divui %flat_a, %m_t_ld : index
      %m_slot = arith.remui %flat_a, %m_t_ld : index
      %m_off = affine.apply affine_map<(ms)[mb] -> ((mb + ms) * 16)>(%m_slot)[%m_global_base]
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      %fut = func.call @load_global_tile_16x64_b(%A_ptr, %m_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index, index, index) -> !future_global_read
      memref.store %fut, %gfut_a[%i] : !gfut_a_buf
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue dwordx4 global loads for a flat range of B tiles.
  func.func private @k_load_b_flat(%b_per_wave: index, %k_t: index, %n_t_ld: index,
      %B_ptr: !aster_utils.any, %k: index, %stride_AB: index,
      %n_global_base: index, %flat_b_start: index) -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %gfut_b = memref.alloca(%b_per_wave) : !gfut_b_buf
    scf.for %i = %c0 to %b_per_wave step %c1 {
      %flat_b = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_b_start]
      %kt = arith.divui %flat_b, %n_t_ld : index
      %n_slot = arith.remui %flat_b, %n_t_ld : index
      %n_off = affine.apply affine_map<(ns)[nb] -> ((nb + ns) * 16)>(%n_slot)[%n_global_base]
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      %fut = func.call @load_global_tile_16x64_b(%B_ptr, %n_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index, index, index) -> !future_global_read
      memref.store %fut, %gfut_b[%i] : !gfut_b_buf
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Issue dwordx4 global loads for a flat range of A tiles into a pre-allocated buffer.
  // Same as @k_load_a_flat but writes into %out_buf instead of allocating a new one.
  // Used to overlap global loads for k+1 with the MFMA compute phase for k.
  func.func private @k_load_a_flat_into(%a_per_wave: index, %k_t: index, %m_t_ld: index,
      %A_ptr: !aster_utils.any, %k: index, %stride_AB: index,
      %m_global_base: index, %flat_a_start: index, %out_buf: !gfut_a_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %a_per_wave step %c1 {
      %flat_a = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_a_start]
      %kt = arith.divui %flat_a, %m_t_ld : index
      %m_slot = arith.remui %flat_a, %m_t_ld : index
      %m_off = affine.apply affine_map<(ms)[mb] -> ((mb + ms) * 16)>(%m_slot)[%m_global_base]
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      %fut = func.call @load_global_tile_16x64_b(%A_ptr, %m_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index, index, index) -> !future_global_read
      memref.store %fut, %out_buf[%i] : !gfut_a_buf
    } {aster.constexpr}
    return
  }

  // Issue dwordx4 global loads for a flat range of B tiles into a pre-allocated buffer.
  // Same as @k_load_b_flat but writes into %out_buf instead of allocating a new one.
  func.func private @k_load_b_flat_into(%b_per_wave: index, %k_t: index, %n_t_ld: index,
      %B_ptr: !aster_utils.any, %k: index, %stride_AB: index,
      %n_global_base: index, %flat_b_start: index, %out_buf: !gfut_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %b_per_wave step %c1 {
      %flat_b = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_b_start]
      %kt = arith.divui %flat_b, %n_t_ld : index
      %n_slot = arith.remui %flat_b, %n_t_ld : index
      %n_off = affine.apply affine_map<(ns)[nb] -> ((nb + ns) * 16)>(%n_slot)[%n_global_base]
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      %fut = func.call @load_global_tile_16x64_b(%B_ptr, %n_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index, index, index) -> !future_global_read
      memref.store %fut, %out_buf[%i] : !gfut_b_buf
    } {aster.constexpr}
    return
  }

  // Store a flat range of A tiles to LDS.
  // LDS offset: (kt * m_t_ld + m_slot) * 1024 + base_a.
  func.func private @k_store_a_flat(%a_per_wave: index, %k_t: index, %m_t_ld: index,
      %base_a: index, %flat_a_start: index, %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s] -> (s * 2)>()[%a_per_wave]
    %tok_a = memref.alloca(%tok_buf_size) : !tok_a_buf
    scf.for %i = %c0 to %a_per_wave step %c1 {
      %flat_a = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_a_start]
      %kt = arith.divui %flat_a, %m_t_ld : index
      %m_slot = arith.remui %flat_a, %m_t_ld : index
      %off = affine.apply affine_map<(kt, ms)[base, mtld] -> (base + (kt * mtld + ms) * 1024)>
          (%kt, %m_slot)[%base_a, %m_t_ld]
      %gfut = memref.load %gfut_a[%i] : !gfut_a_buf
      %tok_lo, %tok_hi = func.call @store_global_tile_to_lds_16x64_b(%off, %gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_idx_lo = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 * 2 + 1)>(%i)
      memref.store %tok_lo, %tok_a[%tok_idx_lo] : !tok_a_buf
      memref.store %tok_hi, %tok_a[%tok_idx_hi] : !tok_a_buf
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Store a flat range of B tiles to LDS.
  func.func private @k_store_b_flat(%b_per_wave: index, %k_t: index, %n_t_ld: index,
      %base_b: index, %flat_b_start: index, %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s] -> (s * 2)>()[%b_per_wave]
    %tok_b = memref.alloca(%tok_buf_size) : !tok_b_buf
    scf.for %i = %c0 to %b_per_wave step %c1 {
      %flat_b = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_b_start]
      %kt = arith.divui %flat_b, %n_t_ld : index
      %n_slot = arith.remui %flat_b, %n_t_ld : index
      %off = affine.apply affine_map<(kt, ns)[base, ntld] -> (base + (kt * ntld + ns) * 1024)>
          (%kt, %n_slot)[%base_b, %n_t_ld]
      %gfut = memref.load %gfut_b[%i] : !gfut_b_buf
      %tok_lo, %tok_hi = func.call @store_global_tile_to_lds_16x64_b(%off, %gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_idx_lo = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
      %tok_idx_hi = affine.apply affine_map<(d0) -> (d0 * 2 + 1)>(%i)
      memref.store %tok_lo, %tok_b[%tok_idx_lo] : !tok_b_buf
      memref.store %tok_hi, %tok_b[%tok_idx_hi] : !tok_b_buf
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Read A tiles from LDS for a flat range of output tiles.
  // Wave owns tiles_per_wave output tiles; tile i has m = (flat_c_start + i) / n_t.
  // Buffer size: tiles_per_wave * k_t * 2 (two MFMA K-steps per K tile).
  // Buffer indexed as [i * k_inner + kt * 2 + kh].
  func.func private @k_read_lds_a_flat(%tiles_per_wave: index, %k_t: index, %m_t_ld: index,
      %base_a: index, %flat_c_start: index, %n_t: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %m = arith.divui %flat_c, %n_t : index
      scf.for %kt = %c0 to %k_t step %c1 {
        scf.for %kh = %c0 to %c2 step %c1 {
          %tile_off = affine.apply affine_map<(kt, m)[base, mtld] -> (base + (kt * mtld + m) * 1024)>
              (%kt, %m)[%base_a, %m_t_ld]
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(i, ki)[kinr] -> (i * kinr + ki)>(%i, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_A_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS for a flat range of output tiles.
  // Wave owns tiles_per_wave output tiles; tile i has n = (flat_c_start + i) % n_t.
  // Buffer size: tiles_per_wave * k_t * 2.
  func.func private @k_read_lds_b_flat(%tiles_per_wave: index, %k_t: index, %n_t_ld: index,
      %base_b: index, %flat_c_start: index, %n_t: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %n = arith.remui %flat_c, %n_t : index
      scf.for %kt = %c0 to %k_t step %c1 {
        scf.for %kh = %c0 to %c2 step %c1 {
          %tile_off = affine.apply affine_map<(kt, n)[base, ntld] -> (base + (kt * ntld + n) * 1024)>
              (%kt, %n)[%base_b, %n_t_ld]
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(i, ki)[kinr] -> (i * kinr + ki)>(%i, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_B_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Fused wait + MFMA for flat output tile distribution.
  // Loop order: (i, kh) with i outermost so consecutive MFMAs update different
  // C accumulators, maximizing AGPR reuse and hiding MFMA latency.
  func.func private @k_wait_and_compute_mfmas_flat(%tiles_per_wave: index, %k_inner: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    scf.for %idx = %c0 to %ub step %c1 {
      %i, %kh = affine.delinearize_index %idx into (%tiles_per_wave, %k_inner) : index, index
      %ak_idx = affine.linearize_index [%i, %kh] by (%tiles_per_wave, %k_inner) : index
      %fut_a_val = memref.load %a_fut[%ak_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %fut_b_val = memref.load %b_fut[%ak_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      %c_old = memref.load %c_buf[%i] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%i] : !c_buf
    } {aster.constexpr}
    return
  }

  // Store flat range of output tiles to global C.
  // Tile i: m = (flat_c_start + i) / n_t, n = (flat_c_start + i) % n_t.
  // Global offsets: m_abs = (m_wg * m_t + m) * 16, n_abs = (n_wg * n_t + n) * 16.
  func.func private @store_c_tiles_flat(%tiles_per_wave: index,
      %c_buf: !c_buf, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_wg: index, %n_wg: index, %m_t: index, %n_t: index, %flat_c_start: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %m = arith.divui %flat_c, %n_t : index
      %n = arith.remui %flat_c, %n_t : index
      %m_abs = affine.apply affine_map<(m)[mwg, mt] -> ((mwg * mt + m) * 16)>(%m)[%m_wg, %m_t]
      %n_abs = affine.apply affine_map<(n)[nwg, nt] -> ((nwg * nt + n) * 16)>(%n)[%n_wg, %n_t]
      %c_tile = memref.load %c_buf[%i] : !c_buf
      // Store tile; no explicit wait needed -- s_endpgm drains all outstanding stores.
      func.call @store_global_C_mfma_f32_16x16x16_f16(%c_tile, %C_ptr, %m_abs, %n_abs, %stride_C)
          : (!rt_C_f32, !aster_utils.any, index, index, index) -> ()
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 7. 2D compute wave distribution helpers (flat loading, 2D LDS-read/compute)
  //===--------------------------------------------------------------------===//
  // Wave (wm, wn) reads A rows [wm*m_t_per_wave, (wm+1)*m_t_per_wave) and
  // B cols [wn*n_t_per_wave, (wn+1)*n_t_per_wave) from LDS, then computes the
  // m_t_per_wave x n_t_per_wave rectangle of output tiles.
  // Loading still uses the flat helpers above; only LDS-read/compute/store are 2D.

  // Read m_t_per_wave A rows from LDS for wave wm.
  // buf_size = m_t_per_wave * k_inner  (k_inner = k_t * 2 for 16x16).
  // buf[m * k_inner + kt*2 + kh] = A row (wm*m_t_per_wave + m), k-step (kt, kh).
  func.func private @k_read_lds_a_2d(%m_t_per_wave: index, %k_t: index, %m_t_ld: index,
      %base_a: index, %wm: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[mtpw, ki] -> (mtpw * ki)>()[%m_t_per_wave, %k_inner]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    %wm_base = affine.apply affine_map<(wm)[mtpw] -> (wm * mtpw)>(%wm)[%m_t_per_wave]
    scf.for %m = %c0 to %m_t_per_wave step %c1 {
      %m_abs = affine.apply affine_map<(m)[wb] -> (wb + m)>(%m)[%wm_base]
      scf.for %kt = %c0 to %k_t step %c1 {
        scf.for %kh = %c0 to %c2 step %c1 {
          %tile_off = affine.apply affine_map<(kt, mabs)[base, mtld] -> (base + (kt * mtld + mabs) * 1024)>
              (%kt, %m_abs)[%base_a, %m_t_ld]
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(m, ki)[kinr] -> (m * kinr + ki)>(%m, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_A_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read n_t_per_wave B cols from LDS for wave wn.
  // buf_size = n_t_per_wave * k_inner.
  // buf[n * k_inner + kt*2 + kh] = B col (wn*n_t_per_wave + n), k-step (kt, kh).
  func.func private @k_read_lds_b_2d(%n_t_per_wave: index, %k_t: index, %n_t_ld: index,
      %base_b: index, %wn: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[ntpw, ki] -> (ntpw * ki)>()[%n_t_per_wave, %k_inner]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    %wn_base = affine.apply affine_map<(wn)[ntpw] -> (wn * ntpw)>(%wn)[%n_t_per_wave]
    scf.for %n = %c0 to %n_t_per_wave step %c1 {
      %n_abs = affine.apply affine_map<(n)[wb] -> (wb + n)>(%n)[%wn_base]
      scf.for %kt = %c0 to %k_t step %c1 {
        scf.for %kh = %c0 to %c2 step %c1 {
          %tile_off = affine.apply affine_map<(kt, nabs)[base, ntld] -> (base + (kt * ntld + nabs) * 1024)>
              (%kt, %n_abs)[%base_b, %n_t_ld]
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 32)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 2 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(n, ki)[kinr] -> (n * kinr + ki)>(%n, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_B_swizzled(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Fused wait + MFMA for 2D output tile rectangle (m_t_per_wave x n_t_per_wave).
  // Loop order: (m, n, kh) with m outermost so consecutive MFMAs over n columns
  // reuse the same A register, maximising AGPR reuse and hiding MFMA latency.
  // a_fut[m*k_inner + kh], b_fut[n*k_inner + kh], C_buf[m*n_t_per_wave + n].
  func.func private @k_wait_and_compute_mfmas_2d(%m_t_per_wave: index, %n_t_per_wave: index,
      %k_inner: index, %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[mtpw, ntpw, ki] -> (mtpw * ntpw * ki)>()
        [%m_t_per_wave, %n_t_per_wave, %k_inner]
    scf.for %idx = %c0 to %ub step %c1 {
      %m, %n, %kh = affine.delinearize_index %idx into
          (%m_t_per_wave, %n_t_per_wave, %k_inner) : index, index, index
      %a_idx = affine.apply affine_map<(m, kh)[ki] -> (m * ki + kh)>(%m, %kh)[%k_inner]
      %b_idx = affine.apply affine_map<(n, kh)[ki] -> (n * ki + kh)>(%n, %kh)[%k_inner]
      %c_idx = affine.apply affine_map<(m, n)[ntpw] -> (m * ntpw + n)>(%m, %n)[%n_t_per_wave]
      %fut_a_val = memref.load %a_fut[%a_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      %fut_b_val = memref.load %b_fut[%b_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }

  // Fused LDS read + MFMA compute with intra-step prefetch for the 2D output
  // tile rectangle (m_t_per_wave × n_t_per_wave). Issues reads for ki+1 before
  // resolving ki so only m_t_per_wave + n_t_per_wave reads are in-flight at
  // each s_waitcnt, instead of k_inner * (m_t_per_wave + n_t_per_wave). This
  // eliminates the lgkmcnt staircase stall in the unprefetched version and
  // closes the remaining ~6-8% efficiency gap from theoretical peak.
  // Replaces k_read_lds_a_2d + k_read_lds_b_2d + k_wait_and_compute_mfmas_2d.
  func.func private @k_fused_lds_read_compute_2d(
      %m_t_per_wave: index, %n_t_per_wave: index,
      %k_t: index, %m_t_ld: index, %n_t_ld: index,
      %base_a: index, %base_b: index,
      %wm: index, %wn: index,
      %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %wm_base = affine.apply affine_map<(wm)[mtpw] -> (wm * mtpw)>(%wm)[%m_t_per_wave]
    %wn_base = affine.apply affine_map<(wn)[ntpw] -> (wn * ntpw)>(%wn)[%n_t_per_wave]

    // Double-buffer: cur holds futures for the current ki step, next for ki+1.
    %a_cur  = memref.alloca(%m_t_per_wave) : !fut_a_buf
    %b_cur  = memref.alloca(%n_t_per_wave) : !fut_b_buf
    %a_next = memref.alloca(%m_t_per_wave) : !fut_a_buf
    %b_next = memref.alloca(%n_t_per_wave) : !fut_b_buf

    // Prologue: issue reads for ki=0 (kt=0, k_byte_offset=0).
    scf.for %m = %c0 to %m_t_per_wave step %c1 {
      %m_abs = affine.apply affine_map<(m)[wb] -> (wb + m)>(%m)[%wm_base]
      %tile_off = affine.apply affine_map<(mabs)[base] -> (base + mabs * 1024)>
          (%m_abs)[%base_a]
      %fut = func.call @load_lds_A_swizzled(%tile_off, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index, index, index) -> !future_lds_read
      memref.store %fut, %a_cur[%m] : !fut_a_buf
    } {aster.constexpr}
    scf.for %n = %c0 to %n_t_per_wave step %c1 {
      %n_abs = affine.apply affine_map<(n)[wb] -> (wb + n)>(%n)[%wn_base]
      %tile_off = affine.apply affine_map<(nabs)[base] -> (base + nabs * 1024)>
          (%n_abs)[%base_b]
      %fut = func.call @load_lds_B_swizzled(%tile_off, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32}
          : (index, index, index) -> !future_lds_read
      memref.store %fut, %b_cur[%n] : !fut_b_buf
    } {aster.constexpr}

    // Main loop: for each ki step, prefetch ki+1 then resolve ki and compute.
    scf.for %ki = %c0 to %k_inner step %c1 {
      %ki1 = arith.addi %ki, %c1 : index
      %not_last = arith.cmpi slt, %ki1, %k_inner : index
      // Issue reads for ki+1 before resolving ki to minimise s_waitcnt stall.
      scf.if %not_last {
        %kt_next, %half_next = affine.delinearize_index %ki1 into (%k_t, %c2)
            : index, index
        %k_byte_next = affine.apply affine_map<(h) -> (h * 32)>(%half_next)
        scf.for %m = %c0 to %m_t_per_wave step %c1 {
          %m_abs = affine.apply affine_map<(m)[wb] -> (wb + m)>(%m)[%wm_base]
          %tile_off = affine.apply
              affine_map<(kt, mabs)[base, mtld] -> (base + (kt * mtld + mabs) * 1024)>
              (%kt_next, %m_abs)[%base_a, %m_t_ld]
          %fut = func.call @load_lds_A_swizzled(%tile_off, %k_byte_next, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_next[%m] : !fut_a_buf
        } {aster.constexpr}
        scf.for %n = %c0 to %n_t_per_wave step %c1 {
          %n_abs = affine.apply affine_map<(n)[wb] -> (wb + n)>(%n)[%wn_base]
          %tile_off = affine.apply
              affine_map<(kt, nabs)[base, ntld] -> (base + (kt * ntld + nabs) * 1024)>
              (%kt_next, %n_abs)[%base_b, %n_t_ld]
          %fut = func.call @load_lds_B_swizzled(%tile_off, %k_byte_next, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_next[%n] : !fut_b_buf
        } {aster.constexpr}
      }
      // Resolve cur reads then compute m_t_per_wave × n_t_per_wave MFMAs.
      // Loop order: m outermost so each a_val is reused across all n iterations.
      scf.for %m = %c0 to %m_t_per_wave step %c1 {
        %fut_a = memref.load %a_cur[%m] : !fut_a_buf
        %a_val = func.call @get_lds_read_value_vx2(%fut_a)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_lds_read) -> !rt_A_f16
        scf.for %n = %c0 to %n_t_per_wave step %c1 {
          %fut_b = memref.load %b_cur[%n] : !fut_b_buf
          %b_val = func.call @get_lds_read_value_vx2(%fut_b)
              {sched.stage = {{STAGE_COMPUTE}} : i32}
              : (!future_lds_read) -> !rt_B_f16
          %c_idx = affine.apply affine_map<(m, n)[ntpw] -> (m * ntpw + n)>(%m, %n)[%n_t_per_wave]
          %c_old = memref.load %c_buf[%c_idx] : !c_buf
          %c_new = func.call @mfma_f32_16x16x16_f16(%a_val, %b_val, %c_old)
              {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
              : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
          memref.store %c_new, %c_buf[%c_idx] : !c_buf
        } {aster.constexpr}
      } {aster.constexpr}
      // Advance cur <- next for the following ki iteration.
      scf.for %m = %c0 to %m_t_per_wave step %c1 {
        %fut = memref.load %a_next[%m] : !fut_a_buf
        memref.store %fut, %a_cur[%m] : !fut_a_buf
      } {aster.constexpr}
      scf.for %n = %c0 to %n_t_per_wave step %c1 {
        %fut = memref.load %b_next[%n] : !fut_b_buf
        memref.store %fut, %b_cur[%n] : !fut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Store 2D rectangle of output tiles to global C.
  // Wave (wm, wn) owns tiles m in [0, m_t_per_wave), n in [0, n_t_per_wave).
  // Global offsets: m_abs = (m_wg*m_t + wm*m_t_per_wave + m)*16, n_abs similar.
  func.func private @store_c_tiles_2d(%m_t_per_wave: index, %n_t_per_wave: index,
      %c_buf: !c_buf, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_wg: index, %n_wg: index, %m_t: index, %n_t: index, %wm: index, %wn: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[mtpw, ntpw] -> (mtpw * ntpw)>()
        [%m_t_per_wave, %n_t_per_wave]
    // Pre-compute per-wave M/N base offsets in MFMA-tile units.
    %block_m_base = affine.apply affine_map<(mwg)[mt] -> (mwg * mt)>(%m_wg)[%m_t]
    %wave_m_base  = affine.apply affine_map<(wm)[mtpw] -> (wm * mtpw)>(%wm)[%m_t_per_wave]
    %m_base = arith.addi %block_m_base, %wave_m_base : index
    %block_n_base = affine.apply affine_map<(nwg)[nt] -> (nwg * nt)>(%n_wg)[%n_t]
    %wave_n_base  = affine.apply affine_map<(wn)[ntpw] -> (wn * ntpw)>(%wn)[%n_t_per_wave]
    %n_base = arith.addi %block_n_base, %wave_n_base : index
    scf.for %i = %c0 to %ub step %c1 {
      %m, %n = affine.delinearize_index %i into (%m_t_per_wave, %n_t_per_wave) : index, index
      %c_idx = affine.apply affine_map<(m, n)[ntpw] -> (m * ntpw + n)>(%m, %n)[%n_t_per_wave]
      %m_abs = affine.apply affine_map<(m)[mb] -> ((mb + m) * 16)>(%m)[%m_base]
      %n_abs = affine.apply affine_map<(n)[nb] -> ((nb + n) * 16)>(%n)[%n_base]
      %c_tile = memref.load %c_buf[%c_idx] : !c_buf
      // Store tile; no explicit wait needed -- s_endpgm drains all outstanding stores.
      func.call @store_global_C_mfma_f32_16x16x16_f16(%c_tile, %C_ptr, %m_abs, %n_abs, %stride_C)
          : (!rt_C_f32, !aster_utils.any, index, index, index) -> ()
    } {aster.constexpr}
    return
  }
