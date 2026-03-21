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

  //===--------------------------------------------------------------------===//
  // 7. Flat wave distribution helpers (32x32x8 variant)
  //===--------------------------------------------------------------------===//
  // Flat loading is over (k_t * m_t_ld) 16-row half-tile slots, where
  // m_t_ld = m_tile / 16 = 2 * m_t.  Each flat index maps to one 16x64_b load.
  // LDS layout per half-tile slot: (kt * m_t_ld + m_slot) * 1024 + base.
  // LDS read combines two consecutive half-tile slots (2*m, 2*m+1) into one
  // 32-row MFMA tile using @load_lds_A_swizzled_32x32.

  // Issue dwordx4 global loads for a flat range of A half-tiles (16-row units).
  func.func private @k_load_a_flat_32x32(%a_per_wave: index, %k_t: index, %m_t_ld: index,
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

  // Issue dwordx4 global loads for a flat range of B half-tiles (16-row units).
  func.func private @k_load_b_flat_32x32(%b_per_wave: index, %k_t: index, %n_t_ld: index,
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

  // Store a flat range of A half-tiles to LDS.
  // LDS offset: (kt * m_t_ld + m_slot) * 1024 + base_a.
  func.func private @k_store_a_flat_32x32(%a_per_wave: index, %k_t: index, %m_t_ld: index,
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

  // Store a flat range of B half-tiles to LDS.
  func.func private @k_store_b_flat_32x32(%b_per_wave: index, %k_t: index, %n_t_ld: index,
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

  // Read A tiles from LDS for flat output tile range (32x32 variant).
  // Tile i (in 32-row MFMA units): m = (flat_c_start + i) / n_t.
  // LDS tile_off points to the upper half-tile: (kt * m_t_ld + 2*m) * 1024 + base.
  // @load_lds_A_swizzled_32x32 reads both halves (32 rows) starting at tile_off.
  // k_byte_offset = kh * 16 (8 f16 per MFMA K-step * 2 bytes, 4 steps per K_T).
  // Buffer size: tiles_per_wave * k_t * 4 (= tiles_per_wave * k_inner).
  func.func private @k_read_lds_a_flat_32x32(%tiles_per_wave: index, %k_t: index, %m_t_ld: index,
      %base_a: index, %flat_c_start: index, %n_t: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %m = arith.divui %flat_c, %n_t : index
      scf.for %kt = %c0 to %k_t step %c1 {
        // tile_off: upper half-tile of the 32-row MFMA tile m at K-step kt.
        %tile_off = affine.apply affine_map<(kt, m)[base, mtld] -> (base + (kt * mtld + m * 2) * 1024)>
            (%kt, %m)[%base_a, %m_t_ld]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(i, ki)[kinr] -> (i * kinr + ki)>(%i, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_A_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read B tiles from LDS for flat output tile range (32x32 variant).
  // n = (flat_c_start + i) % n_t (32-row MFMA tile index).
  func.func private @k_read_lds_b_flat_32x32(%tiles_per_wave: index, %k_t: index, %n_t_ld: index,
      %base_b: index, %flat_c_start: index, %n_t: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %n = arith.remui %flat_c, %n_t : index
      scf.for %kt = %c0 to %k_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, n)[base, ntld] -> (base + (kt * ntld + n * 2) * 1024)>
            (%kt, %n)[%base_b, %n_t_ld]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(i, ki)[kinr] -> (i * kinr + ki)>(%i, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_B_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Fused wait + MFMA for flat output tile distribution (32x32x8 variant).
  func.func private @k_wait_and_compute_mfmas_flat_32x32(%tiles_per_wave: index, %k_inner: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf_32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[tpw, ki] -> (tpw * ki)>()[%tiles_per_wave, %k_inner]
    scf.for %idx = %c0 to %ub step %c1 {
      %i, %kh = affine.delinearize_index %idx into (%tiles_per_wave, %k_inner) : index, index
      %ak_idx = affine.linearize_index [%i, %kh] by (%tiles_per_wave, %k_inner) : index
      %fut_a_val = memref.load %a_fut[%ak_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16_32
      %fut_b_val = memref.load %b_fut[%ak_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16_32
      %c_old = memref.load %c_buf[%i] : !c_buf_32
      %c_new = func.call @mfma_f32_32x32x8_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
          : (!rt_A_f16_32, !rt_B_f16_32, !rt_C_f32_32) -> !rt_C_f32_32
      memref.store %c_new, %c_buf[%i] : !c_buf_32
    } {aster.constexpr}
    return
  }

  // Store flat range of output tiles to global C (32x32 variant).
  // Tile i: m = (flat_c_start + i) / n_t, n = (flat_c_start + i) % n_t  (32-row units).
  // Global offsets: m_abs = (m_wg * m_t + m) * 32, n_abs = (n_wg * n_t + n) * 32.
  func.func private @store_c_tiles_flat_32x32(%tiles_per_wave: index,
      %c_buf: !c_buf_32, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_wg: index, %n_wg: index, %m_t: index, %n_t: index, %flat_c_start: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %tiles_per_wave step %c1 {
      %flat_c = affine.apply affine_map<(i)[s] -> (s + i)>(%i)[%flat_c_start]
      %m = arith.divui %flat_c, %n_t : index
      %n = arith.remui %flat_c, %n_t : index
      %m_abs = affine.apply affine_map<(m)[mwg, mt] -> ((mwg * mt + m) * 32)>(%m)[%m_wg, %m_t]
      %n_abs = affine.apply affine_map<(n)[nwg, nt] -> ((nwg * nt + n) * 32)>(%n)[%n_wg, %n_t]
      %c_tile = memref.load %c_buf[%i] : !c_buf_32
      // Store tile; s_endpgm drains all outstanding stores.
      func.call @store_global_C_mfma_f32_32x32x8_f16(%c_tile, %C_ptr, %m_abs, %n_abs, %stride_C)
          : (!rt_C_f32_32, !aster_utils.any, index, index, index) -> ()
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // 8. 2D compute wave distribution helpers (32x32 variant)
  //===--------------------------------------------------------------------===//
  // Same structure as the 16x16 2D helpers, but:
  //   - tile_off = (kt * m_t_ld + m_abs * 2) * 1024 + base  (two 16-row LDS slots per 32-row tile)
  //   - k_inner = k_t * 4, k_byte_offset = kh * 16
  //   - uses @load_lds_A_swizzled_32x32, @mfma_f32_32x32x8_f16, !c_buf_32
  //   - store uses scale factor 32

  // Read m_t_per_wave A rows from LDS for wave wm (32x32 variant).
  // m_t_per_wave counts 32-row MFMA tiles; m_t_ld counts 16-row LDS slots (= 2 * m_t).
  func.func private @k_read_lds_a_2d_32x32(%m_t_per_wave: index, %k_t: index, %m_t_ld: index,
      %base_a: index, %wm: index) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[mtpw, ki] -> (mtpw * ki)>()[%m_t_per_wave, %k_inner]
    %a_fut = memref.alloca(%buf_size) : !fut_a_buf
    %wm_base = affine.apply affine_map<(wm)[mtpw] -> (wm * mtpw)>(%wm)[%m_t_per_wave]
    scf.for %m = %c0 to %m_t_per_wave step %c1 {
      %m_abs = affine.apply affine_map<(m)[wb] -> (wb + m)>(%m)[%wm_base]
      scf.for %kt = %c0 to %k_t step %c1 {
        // tile_off: upper 16-row half-tile of the 32-row MFMA tile m_abs at K-step kt.
        %tile_off = affine.apply affine_map<(kt, mabs)[base, mtld] -> (base + (kt * mtld + mabs * 2) * 1024)>
            (%kt, %m_abs)[%base_a, %m_t_ld]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(m, ki)[kinr] -> (m * kinr + ki)>(%m, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_A_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %a_fut[%buf_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_fut : !fut_a_buf
  }

  // Read n_t_per_wave B cols from LDS for wave wn (32x32 variant).
  func.func private @k_read_lds_b_2d_32x32(%n_t_per_wave: index, %k_t: index, %n_t_ld: index,
      %base_b: index, %wn: index) -> !fut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %k_inner = affine.apply affine_map<()[kt] -> (kt * 4)>()[%k_t]
    %buf_size = affine.apply affine_map<()[ntpw, ki] -> (ntpw * ki)>()[%n_t_per_wave, %k_inner]
    %b_fut = memref.alloca(%buf_size) : !fut_b_buf
    %wn_base = affine.apply affine_map<(wn)[ntpw] -> (wn * ntpw)>(%wn)[%n_t_per_wave]
    scf.for %n = %c0 to %n_t_per_wave step %c1 {
      %n_abs = affine.apply affine_map<(n)[wb] -> (wb + n)>(%n)[%wn_base]
      scf.for %kt = %c0 to %k_t step %c1 {
        %tile_off = affine.apply affine_map<(kt, nabs)[base, ntld] -> (base + (kt * ntld + nabs * 2) * 1024)>
            (%kt, %n_abs)[%base_b, %n_t_ld]
        scf.for %kh = %c0 to %c4 step %c1 {
          %k_byte_offset = affine.apply affine_map<(kh) -> (kh * 16)>(%kh)
          %ki_idx = affine.apply affine_map<(kt, kh) -> (kt * 4 + kh)>(%kt, %kh)
          %buf_idx = affine.apply affine_map<(n, ki)[kinr] -> (n * kinr + ki)>(%n, %ki_idx)[%k_inner]
          %fut = func.call @load_lds_B_swizzled_32x32(%tile_off, %k_byte_offset, %c2)
              {sched.stage = {{STAGE_DS_READ}} : i32}
              : (index, index, index) -> !future_lds_read
          memref.store %fut, %b_fut[%buf_idx] : !fut_b_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %b_fut : !fut_b_buf
  }

  // Fused wait + MFMA for 2D output tile rectangle (32x32x8 variant).
  func.func private @k_wait_and_compute_mfmas_2d_32x32(%m_t_per_wave: index, %n_t_per_wave: index,
      %k_inner: index, %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf_32) {
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
          : (!future_lds_read) -> !rt_A_f16_32
      %fut_b_val = memref.load %b_fut[%b_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b_val)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16_32
      %c_old = memref.load %c_buf[%c_idx] : !c_buf_32
      %c_new = func.call @mfma_f32_32x32x8_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32, sched.rotate_head}
          : (!rt_A_f16_32, !rt_B_f16_32, !rt_C_f32_32) -> !rt_C_f32_32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf_32
    } {aster.constexpr}
    return
  }

  // Store 2D rectangle of output tiles to global C (32x32 variant).
  func.func private @store_c_tiles_2d_32x32(%m_t_per_wave: index, %n_t_per_wave: index,
      %c_buf: !c_buf_32, %C_ptr: !aster_utils.any, %stride_C: index,
      %m_wg: index, %n_wg: index, %m_t: index, %n_t: index, %wm: index, %wn: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[mtpw, ntpw] -> (mtpw * ntpw)>()
        [%m_t_per_wave, %n_t_per_wave]
    %block_m_base = affine.apply affine_map<(mwg)[mt] -> (mwg * mt)>(%m_wg)[%m_t]
    %wave_m_base  = affine.apply affine_map<(wm)[mtpw] -> (wm * mtpw)>(%wm)[%m_t_per_wave]
    %m_base = arith.addi %block_m_base, %wave_m_base : index
    %block_n_base = affine.apply affine_map<(nwg)[nt] -> (nwg * nt)>(%n_wg)[%n_t]
    %wave_n_base  = affine.apply affine_map<(wn)[ntpw] -> (wn * ntpw)>(%wn)[%n_t_per_wave]
    %n_base = arith.addi %block_n_base, %wave_n_base : index
    scf.for %i = %c0 to %ub step %c1 {
      %m, %n = affine.delinearize_index %i into (%m_t_per_wave, %n_t_per_wave) : index, index
      %c_idx = affine.apply affine_map<(m, n)[ntpw] -> (m * ntpw + n)>(%m, %n)[%n_t_per_wave]
      %m_abs = affine.apply affine_map<(m)[mb] -> ((mb + m) * 32)>(%m)[%m_base]
      %n_abs = affine.apply affine_map<(n)[nb] -> ((nb + n) * 32)>(%n)[%n_base]
      %c_tile = memref.load %c_buf[%c_idx] : !c_buf_32
      // Store tile; s_endpgm drains all outstanding stores.
      func.call @store_global_C_mfma_f32_32x32x8_f16(%c_tile, %C_ptr, %m_abs, %n_abs, %stride_C)
          : (!rt_C_f32_32, !aster_utils.any, index, index, index) -> ()
    } {aster.constexpr}
    return
  }
