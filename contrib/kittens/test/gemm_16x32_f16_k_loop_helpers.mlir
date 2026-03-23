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

  // Fused wait + MFMA: delinearize flat index into (m, k, n), wait for
  // the corresponding A[k,m] and B[k,n] futures, then MFMA.
  // Redundant waits on already-completed tokens are free (waitcnt satisfied).
  // k_mfma is the total number of MFMA K-steps (e.g. k_t * 2 for 16x32 tiles).
  func.func private @k_wait_and_compute_mfmas(%m_t: index, %n_t: index, %k_mfma: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %km, %nt = affine.delinearize_index %idx into (%m_t, %k_mfma, %n_t) : index, index, index
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

  // Compute MFMAs: m_t * k_mfma * n_t operations.
  // k_mfma is the total number of MFMA K-steps; caller computes it (e.g. k_t * 2 for 16x32 tiles).
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index, %k_mfma: index,
      %a_vals: !vals_a_buf, %b_vals: !vals_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %km, %nt = affine.delinearize_index %idx into (%m_t, %k_mfma, %n_t) : index, index, index
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
