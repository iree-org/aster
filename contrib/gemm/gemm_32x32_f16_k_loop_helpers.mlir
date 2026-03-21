  // From lds_mfma_32x32_f16.mlir
  func.func private @load_lds_A_swizzled_32x32(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled_32x32(index, index, index) -> !future_lds_read
  func.func private @compute_lds_read_addr_A_32x32(index, index, index) -> index
  func.func private @compute_lds_read_addr_B_32x32(index, index, index) -> index
  // From global_16x64_b.mlir (reused for 32x32: load upper/lower halves separately)
  func.func private @compute_global_byte_off_16x64_b(index, index, index) -> index
  func.func private @load_global_at_byte_off(!aster_utils.any, index) -> !future_global_read
  func.func private @compute_lds_write_addrs_16x64_b(index) -> (index, index)
  func.func private @get_global_load_value_vx4(!future_global_read) -> !vx4
  func.func private @write_vx4_to_lds_at(!vx4, index, index) -> (!lds_write_token, !lds_write_token)
  func.func private @read_vx2_from_lds_at(index) -> !future_lds_read

  // === K-loop helper functions for 32x32x8 MFMA (v_mfma_f32_32x32x8_f16) ===
  //
  // Each 32-row M/N-tile is treated as two consecutive 16x64_b slots in LDS:
  //   upper half (rows  0-15): slot 2*i     at LDS byte offset (... + 2*i    ) * 1024
  //   lower half (rows 16-31): slot 2*i + 1 at LDS byte offset (... + 2*i + 1) * 1024
  //
  // Global loading uses @load_global_tile_16x64_b for each half-tile.
  // LDS addressing passes m_base_16 (in 16-row tile units) directly, so:
  //   m_off_upper = (m_base_16 + 2*i)     * 16  (global row in 16-row units)
  //   m_off_lower = (m_base_16 + 2*i + 1) * 16
  //
  // LDS read uses @load_lds_A/B_swizzled_32x32 with:
  //   k_mfma_total = k_t * 4  (4 MFMA steps per K_T chunk: 32 f16 / 8 = 4)
  //   k_byte_offset = kh * 16  (8 f16 per MFMA step * 2 bytes = 16 bytes)
  //   tile_off = base + (kt * M_T_LD + wave_base_16 + 2*i) * 1024
  //
  // Buffer sizes:
  //   Global load futures:  2 * k_t * m_t_32  (2 per 32-row tile: upper + lower)
  //   LDS write tokens:     4 * k_t * m_t_32  (2 lo/hi tokens per half-tile, 2 halves)
  //   LDS read futures:     k_t * 4 * m_t_32  (k_mfma_total * m_t_32)

  //===--------------------------------------------------------------------===//
  // 1. Global load (per 32-row tile = 2 x 16x64_b loads)
  //===--------------------------------------------------------------------===//

  // Compute global load byte offsets for A tiles (upper + lower halves).
  // m_t_32: number of 32-row A tiles this wave loads.
  // m_base_16: starting 16-row tile index in global A (= wave_a_base in 16-row units).
  // Returns buffer of size 2 * k_t * m_t_32:
  //   [kt, i, half=0] = upper half (rows 0-15 of tile i at k-step kt)
  //   [kt, i, half=1] = lower half (rows 16-31)
  func.func private @k_compute_global_addrs_a_32x32(%m_t_32: index, %k_t: index,
      %k: index, %stride_AB: index, %m_base_16: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %m_t_32]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %m_t_32 step %c1 {
        // Upper half: absolute row = (m_base_16 + 2*i) * 16
        %m_off_upper = affine.apply
            affine_map<(d0)[s0] -> ((s0 + d0 * 2) * 16)>(%i)[%m_base_16]
        // Lower half: absolute row = (m_base_16 + 2*i + 1) * 16
        %m_off_lower = affine.apply
            affine_map<(d0)[s0] -> ((s0 + d0 * 2 + 1) * 16)>(%i)[%m_base_16]
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 2)>(%kt, %i)[%m_t_32]
        %byte_off_upper = func.call @compute_global_byte_off_16x64_b(%m_off_upper, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (index, index, index) -> index
        %byte_off_lower = func.call @compute_global_byte_off_16x64_b(%m_off_lower, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (index, index, index) -> index
        %idx_lower = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        memref.store %byte_off_upper, %addrs[%base_idx] : memref<?xindex>
        memref.store %byte_off_lower, %addrs[%idx_lower] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Compute global load byte offsets for B tiles (upper + lower halves).
  // n_t_32: number of 32-row B tiles this wave loads.
  // n_base_16: starting 16-row tile index in global B.
  func.func private @k_compute_global_addrs_b_32x32(%n_t_32: index, %k_t: index,
      %k: index, %stride_AB: index, %n_base_16: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 2)>()[%k_t, %n_t_32]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      %k_offset = affine.apply affine_map<(kt)[kb] -> ((kb + kt) * 32)>(%kt)[%k]
      scf.for %i = %c0 to %n_t_32 step %c1 {
        // Absolute row = (n_base_16 + 2*i) * 16 for upper, +1 for lower
        %n_off_upper = affine.apply
            affine_map<(d0)[s0] -> ((s0 + d0 * 2) * 16)>(%i)[%n_base_16]
        %n_off_lower = affine.apply
            affine_map<(d0)[s0] -> ((s0 + d0 * 2 + 1) * 16)>(%i)[%n_base_16]
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 2)>(%kt, %i)[%n_t_32]
        %byte_off_upper = func.call @compute_global_byte_off_16x64_b(%n_off_upper, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (index, index, index) -> index
        %byte_off_lower = func.call @compute_global_byte_off_16x64_b(%n_off_lower, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
            : (index, index, index) -> index
        %idx_lower = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        memref.store %byte_off_upper, %addrs[%base_idx] : memref<?xindex>
        memref.store %byte_off_lower, %addrs[%idx_lower] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Issue global loads at pre-computed byte offsets for A tiles (32x32: 2 per tile).
  func.func private @k_issue_global_loads_a_32x32(%addrs: memref<?xindex>,
      %A_ptr: !aster_utils.any) -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %byte_off = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @load_global_at_byte_off(%A_ptr, %byte_off)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index) -> !future_global_read
      memref.store %fut, %gfut_a[%idx] : !gfut_a_buf
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue global loads at pre-computed byte offsets for B tiles (32x32: 2 per tile).
  func.func private @k_issue_global_loads_b_32x32(%addrs: memref<?xindex>,
      %B_ptr: !aster_utils.any) -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = memref.dim %addrs, %c0 : memref<?xindex>
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf
    scf.for %idx = %c0 to %buf_size step %c1 {
      %byte_off = memref.load %addrs[%idx] : memref<?xindex>
      %fut = func.call @load_global_at_byte_off(%B_ptr, %byte_off)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!aster_utils.any, index) -> !future_global_read
      memref.store %fut, %gfut_b[%idx] : !gfut_b_buf
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Convenience: compute addresses and issue loads for A in one call.
  func.func private @k_load_a_32x32_from_global(%m_t_32: index, %k_t: index,
      %A_ptr: !aster_utils.any, %k: index, %stride_AB: index, %m_base_16: index)
      -> !gfut_a_buf {
    %addrs = func.call @k_compute_global_addrs_a_32x32(%m_t_32, %k_t, %k, %stride_AB, %m_base_16)
        : (index, index, index, index, index) -> memref<?xindex>
    %gfut_a = func.call @k_issue_global_loads_a_32x32(%addrs, %A_ptr)
        : (memref<?xindex>, !aster_utils.any) -> !gfut_a_buf
    return %gfut_a : !gfut_a_buf
  }

  // Convenience: compute addresses and issue loads for B in one call.
  func.func private @k_load_b_32x32_from_global(%n_t_32: index, %k_t: index,
      %B_ptr: !aster_utils.any, %k: index, %stride_AB: index, %n_base_16: index)
      -> !gfut_b_buf {
    %addrs = func.call @k_compute_global_addrs_b_32x32(%n_t_32, %k_t, %k, %stride_AB, %n_base_16)
        : (index, index, index, index, index) -> memref<?xindex>
    %gfut_b = func.call @k_issue_global_loads_b_32x32(%addrs, %B_ptr)
        : (memref<?xindex>, !aster_utils.any) -> !gfut_b_buf
    return %gfut_b : !gfut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 2. LDS write (upper + lower half per 32-row tile)
  //===--------------------------------------------------------------------===//

  // Compute LDS write address pairs for A tiles (32x32: 2 half-tile slots per tile).
  // m_t_32: 32-row tiles to store.
  // wave_a_base_16: starting 16-row LDS slot for this wave (in 16-row slot units).
  // M_T_LD: total 16-row LDS slots per WG per k-chunk (= m_tile / 16).
  // Returns buffer of size 4 * k_t * m_t_32: 2 lo/hi address pairs per half-tile.
  func.func private @k_compute_lds_write_addrs_a_32x32(%m_t_32: index, %k_t: index,
      %base_a: index, %wave_a_base_16: index, %M_T_LD: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t_32]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t_32 step %c1 {
        // Upper half: slot index = kt * M_T_LD + wave_a_base_16 + 2*i
        %slot_upper = affine.apply
            affine_map<(kt, i)[base, wab, mtld] -> (base + (kt * mtld + wab + i * 2) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base_16, %M_T_LD]
        // Lower half: slot index = kt * M_T_LD + wave_a_base_16 + 2*i + 1
        %slot_lower = affine.apply
            affine_map<(kt, i)[base, wab, mtld] -> (base + (kt * mtld + wab + i * 2 + 1) * 1024)>
            (%kt, %i)[%base_a, %wave_a_base_16, %M_T_LD]
        %addr_lo_u, %addr_hi_u = func.call @compute_lds_write_addrs_16x64_b(%slot_upper)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index) -> (index, index)
        %addr_lo_l, %addr_hi_l = func.call @compute_lds_write_addrs_16x64_b(%slot_lower)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index) -> (index, index)
        // Layout in addrs buffer: 4 entries per (kt, i): [lo_upper, hi_upper, lo_lower, hi_lower]
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 4)>(%kt, %i)[%m_t_32]
        %idx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        %idx2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%base_idx)
        %idx3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%base_idx)
        memref.store %addr_lo_u, %addrs[%base_idx] : memref<?xindex>
        memref.store %addr_hi_u, %addrs[%idx1] : memref<?xindex>
        memref.store %addr_lo_l, %addrs[%idx2] : memref<?xindex>
        memref.store %addr_hi_l, %addrs[%idx3] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Compute LDS write address pairs for B tiles (32x32).
  func.func private @k_compute_lds_write_addrs_b_32x32(%n_t_32: index, %k_t: index,
      %base_b: index, %wave_b_base_16: index, %N_T_LD: index) -> memref<?xindex> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t_32]
    %addrs = memref.alloca(%buf_size) : memref<?xindex>
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t_32 step %c1 {
        %slot_upper = affine.apply
            affine_map<(kt, i)[base, wbb, ntld] -> (base + (kt * ntld + wbb + i * 2) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base_16, %N_T_LD]
        %slot_lower = affine.apply
            affine_map<(kt, i)[base, wbb, ntld] -> (base + (kt * ntld + wbb + i * 2 + 1) * 1024)>
            (%kt, %i)[%base_b, %wave_b_base_16, %N_T_LD]
        %addr_lo_u, %addr_hi_u = func.call @compute_lds_write_addrs_16x64_b(%slot_upper)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index) -> (index, index)
        %addr_lo_l, %addr_hi_l = func.call @compute_lds_write_addrs_16x64_b(%slot_lower)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (index) -> (index, index)
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 4)>(%kt, %i)[%n_t_32]
        %idx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        %idx2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%base_idx)
        %idx3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%base_idx)
        memref.store %addr_lo_u, %addrs[%base_idx] : memref<?xindex>
        memref.store %addr_hi_u, %addrs[%idx1] : memref<?xindex>
        memref.store %addr_lo_l, %addrs[%idx2] : memref<?xindex>
        memref.store %addr_hi_l, %addrs[%idx3] : memref<?xindex>
      } {aster.constexpr}
    } {aster.constexpr}
    return %addrs : memref<?xindex>
  }

  // Wait for global loads and write A tiles to LDS at pre-computed addresses.
  // gfut_a: size 2 * k_t * m_t_32 (upper+lower futures interleaved).
  // addrs: size 4 * k_t * m_t_32 (lo_upper, hi_upper, lo_lower, hi_lower per tile).
  func.func private @k_store_to_lds_at_addrs_a_32x32(%addrs: memref<?xindex>,
      %m_t_32: index, %k_t: index, %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %m_t_32]
    %tok_a = memref.alloca(%tok_buf_size) : !tok_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t_32 step %c1 {
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 4)>(%kt, %i)[%m_t_32]
        %gfut_base = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 2)>(%kt, %i)[%m_t_32]
        %idx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        %idx2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%base_idx)
        %idx3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%base_idx)
        %gidx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%gfut_base)
        // Write upper half (gfut index 0 = upper, 1 = lower)
        %gfut_upper = memref.load %gfut_a[%gfut_base] : !gfut_a_buf
        %loaded_upper = func.call @get_global_load_value_vx4(%gfut_upper)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %addr_lo_u = memref.load %addrs[%base_idx] : memref<?xindex>
        %addr_hi_u = memref.load %addrs[%idx1] : memref<?xindex>
        %tok_lo_u, %tok_hi_u = func.call @write_vx4_to_lds_at(%loaded_upper, %addr_lo_u, %addr_hi_u)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo_u, %tok_a[%base_idx] : !tok_a_buf
        memref.store %tok_hi_u, %tok_a[%idx1] : !tok_a_buf
        // Write lower half
        %gfut_lower = memref.load %gfut_a[%gidx1] : !gfut_a_buf
        %loaded_lower = func.call @get_global_load_value_vx4(%gfut_lower)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %addr_lo_l = memref.load %addrs[%idx2] : memref<?xindex>
        %addr_hi_l = memref.load %addrs[%idx3] : memref<?xindex>
        %tok_lo_l, %tok_hi_l = func.call @write_vx4_to_lds_at(%loaded_lower, %addr_lo_l, %addr_hi_l)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo_l, %tok_a[%idx2] : !tok_a_buf
        memref.store %tok_hi_l, %tok_a[%idx3] : !tok_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_a : !tok_a_buf
  }

  // Wait for global loads and write B tiles to LDS at pre-computed addresses.
  func.func private @k_store_to_lds_at_addrs_b_32x32(%addrs: memref<?xindex>,
      %n_t_32: index, %k_t: index, %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1 * 4)>()[%k_t, %n_t_32]
    %tok_b = memref.alloca(%tok_buf_size) : !tok_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t_32 step %c1 {
        %base_idx = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 4)>(%kt, %i)[%n_t_32]
        %gfut_base = affine.apply affine_map<(kt, i)[s] -> ((kt * s + i) * 2)>(%kt, %i)[%n_t_32]
        %idx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%base_idx)
        %idx2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%base_idx)
        %idx3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%base_idx)
        %gidx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%gfut_base)
        %gfut_upper = memref.load %gfut_b[%gfut_base] : !gfut_b_buf
        %loaded_upper = func.call @get_global_load_value_vx4(%gfut_upper)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %addr_lo_u = memref.load %addrs[%base_idx] : memref<?xindex>
        %addr_hi_u = memref.load %addrs[%idx1] : memref<?xindex>
        %tok_lo_u, %tok_hi_u = func.call @write_vx4_to_lds_at(%loaded_upper, %addr_lo_u, %addr_hi_u)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo_u, %tok_b[%base_idx] : !tok_b_buf
        memref.store %tok_hi_u, %tok_b[%idx1] : !tok_b_buf
        %gfut_lower = memref.load %gfut_b[%gidx1] : !gfut_b_buf
        %loaded_lower = func.call @get_global_load_value_vx4(%gfut_lower)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!future_global_read) -> !vx4
        %addr_lo_l = memref.load %addrs[%idx2] : memref<?xindex>
        %addr_hi_l = memref.load %addrs[%idx3] : memref<?xindex>
        %tok_lo_l, %tok_hi_l = func.call @write_vx4_to_lds_at(%loaded_lower, %addr_lo_l, %addr_hi_l)
            {sched.stage = {{STAGE_DS_WRITE}} : i32}
            : (!vx4, index, index) -> (!lds_write_token, !lds_write_token)
        memref.store %tok_lo_l, %tok_b[%idx2] : !tok_b_buf
        memref.store %tok_hi_l, %tok_b[%idx3] : !tok_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %tok_b : !tok_b_buf
  }

  // Convenience: compute LDS write addresses and store A tiles in one call.
  func.func private @k_store_a_32x32_to_lds(%m_t_32: index, %k_t: index,
      %base_a: index, %wave_a_base_16: index, %M_T_LD: index,
      %gfut_a: !gfut_a_buf) -> !tok_a_buf {
    %addrs = func.call @k_compute_lds_write_addrs_a_32x32(%m_t_32, %k_t, %base_a, %wave_a_base_16, %M_T_LD)
        : (index, index, index, index, index) -> memref<?xindex>
    %tok_a = func.call @k_store_to_lds_at_addrs_a_32x32(%addrs, %m_t_32, %k_t, %gfut_a)
        : (memref<?xindex>, index, index, !gfut_a_buf) -> !tok_a_buf
    return %tok_a : !tok_a_buf
  }

  // Convenience: compute LDS write addresses and store B tiles in one call.
  func.func private @k_store_b_32x32_to_lds(%n_t_32: index, %k_t: index,
      %base_b: index, %wave_b_base_16: index, %N_T_LD: index,
      %gfut_b: !gfut_b_buf) -> !tok_b_buf {
    %addrs = func.call @k_compute_lds_write_addrs_b_32x32(%n_t_32, %k_t, %base_b, %wave_b_base_16, %N_T_LD)
        : (index, index, index, index, index) -> memref<?xindex>
    %tok_b = func.call @k_store_to_lds_at_addrs_b_32x32(%addrs, %n_t_32, %k_t, %gfut_b)
        : (memref<?xindex>, index, index, !gfut_b_buf) -> !tok_b_buf
    return %tok_b : !tok_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 3. LDS write wait
  //===--------------------------------------------------------------------===//

  // Wait all LDS write tokens in buf (size determined from memref.dim).
  func.func private @k_wait_lds_writes_32x32(%tok_buf: memref<?x!lds_write_token>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_count = memref.dim %tok_buf, %c0 : memref<?x!lds_write_token>
    scf.for %idx = %c0 to %tok_count step %c1 {
      %tok = memref.load %tok_buf[%idx] : memref<?x!lds_write_token>
      amdgcn.wait deps %tok {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 4. LDS read (k_mfma = k_t * 4 steps)
  //===--------------------------------------------------------------------===//

  // Read A tiles from LDS for 32x32x8 MFMA.
  // k_mfma_total = k_t * 4 (4 MFMA steps per K_T chunk: 32 f16 / 8 = 4).
  // k_byte_offset for step kh = kh * 16 (8 f16 * 2 bytes).
  // tile_off = base + (kt * M_T_LD + wave_a_compute_base_16 + 2*i) * 1024.
  // Buffer indexed as [k_mfma_idx * m_t_32 + i].
  func.func private @k_read_lds_a_32x32(%m_t_32: index, %k_t: index,
      %base_a: index, %wave_a_compute_base_16: index, %M_T_LD: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %m_t_32]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t_32 step %c1 {
        // tile_off = address of the upper half of this 32-row tile in LDS.
        // rows 0-15 at tile_off, rows 16-31 at tile_off + 1024.
        // @load_lds_A_swizzled_32x32 uses @lds_swizzled_addr_32x32 with row 0..31,
        // which correctly spans both halves.
        %tile_off = affine.apply
            affine_map<(kt, i)[base, wac, mtld] -> (base + (kt * mtld + wac + i * 2) * 1024)>
            (%kt, %i)[%base_a, %wave_a_compute_base_16, %M_T_LD]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %m_t_32) : index
          %fut = func.call @load_lds_A_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS for 32x32x8 MFMA.
  func.func private @k_read_lds_b_32x32(%n_t_32: index, %k_t: index,
      %base_b: index, %wave_b_compute_base_16: index, %N_T_LD: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_mfma_total, %n_t_32]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t_32 step %c1 {
        %tile_off = affine.apply
            affine_map<(kt, i)[base, wbc, ntld] -> (base + (kt * ntld + wbc + i * 2) * 1024)>
            (%kt, %i)[%base_b, %wave_b_compute_base_16, %N_T_LD]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %k_mfma_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.linearize_index [%k_mfma_idx, %i] by (%k_mfma_total, %n_t_32) : index
          %fut = func.call @load_lds_B_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  //===--------------------------------------------------------------------===//
  // 5. LDS read wait + MFMA (32x32x8, ax16 accumulators)
  //===--------------------------------------------------------------------===//

  // Fused wait + MFMA: delinearize flat index into (k, m, n), wait for
  // A[k,m] and B[k,n] futures, then call v_mfma_f32_32x32x8_f16.
  // Loop order (k, m, n): keeps all n-tiles for a fixed m updating together,
  // hiding MFMA latency on a given accumulator via intervening MFMAs on others.
  // k_mfma = k_t * 4.
  func.func private @k_wait_and_compute_mfmas_32x32(%m_t_32: index, %n_t_32: index,
      %k_mfma: index, %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf_32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t_32, %k_mfma, %n_t_32]
    scf.for %idx = %c0 to %ub step %c1 {
      %km, %mt, %nt = affine.delinearize_index %idx into (%k_mfma, %m_t_32, %n_t_32)
          : index, index, index
      %a_idx = affine.linearize_index [%km, %mt] by (%k_mfma, %m_t_32) : index
      %b_idx = affine.linearize_index [%km, %nt] by (%k_mfma, %n_t_32) : index

      // Wait + extract A and B (redundant waits on completed tokens are no-ops).
      %fut_a_val = memref.load %a_fut[%a_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16_32
      %fut_b_val = memref.load %b_fut[%b_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16_32

      // MFMA: 32x32x8 with AGPR accumulator (ax16).
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t_32, %n_t_32) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf_32
      %c_new = func.call @mfma_f32_32x32x8_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
          : (!rt_A_f16_32, !rt_B_f16_32, !rt_C_f32_32) -> !rt_C_f32_32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf_32
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 6. Global store epilogue (32x32 output tiles)
  //===--------------------------------------------------------------------===//

  // Store C accumulator tiles to global memory (32x32 output tiles).
  // m_base/n_base: WG's starting tile indices in 32-row/col units.
  func.func private @store_c_tiles_32x32(%m_t_32: index, %n_t_32: index,
      %c_buf: !c_buf_32, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_base: index, %n_base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t_32 step %c1 {
      scf.for %nt = %c0 to %n_t_32 step %c1 {
        %idx = affine.linearize_index [%mt, %nt] by (%m_t_32, %n_t_32) : index
        %c_tile = memref.load %c_buf[%idx] : !c_buf_32
        %m_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%mt)[%m_base]
        %n_off = affine.apply affine_map<(d0)[s0] -> ((s0 + d0) * 32)>(%nt)[%n_base]
        // Store tile; s_endpgm drains all outstanding stores.
        func.call @store_global_C_mfma_f32_32x32x8_f16(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32_32, !aster_utils.any, index, index, index) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }
