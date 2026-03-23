  // Direct-AB helpers: BOTH A and B bypass LDS via MFMA-aware preshuffle.
  // Zero LDS usage. Both loaded directly from global into registers.
  // Same preshuffle layout for A and B (MFMA A and B have identical fragment maps).

  func.func private @lane_id() -> index

  // Issue A global loads at preshuffle byte offsets.
  // Same layout as B: [M_BLOCKS, K_BLOCKS, 64] : [STRIDE_M0_BYTES, 1024, 16]
  func.func private @k_load_a_direct(%m_t: index, %k_t: index,
      %A_ptr: !aster_utils.any, %k: index, %m_base: index)
      -> !gfut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %m_t]
    %gfut_a = memref.alloca(%buf_size) : !gfut_a_buf

    %lid = func.call @lane_id() : () -> index

    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %m_block = affine.apply affine_map<(mt)[mb] -> (mb + mt)>(%i)[%m_base]
        %k_block = affine.apply affine_map<(kt)[kb] -> (kb + kt)>(%kt)[%k]

        %flat_coord = affine.linearize_index [%m_block, %k_block, %lid]
            by ({{M_BLOCKS}}, {{K_BLOCKS}}, 64) : index

        %a_byte_off = layout.linearize %flat_coord,
            #layout.strided_layout<[{{M_BLOCKS}}, {{K_BLOCKS}}, 64] : [{{STRIDE_M0_BYTES}}, 1024, 16]>

        %idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %fut = func.call @load_global_at_byte_off(%A_ptr, %a_byte_off)
            {sched.stage = {{A_STAGE_LOAD}} : i32}
            : (!aster_utils.any, index) -> !future_global_read
        memref.store %fut, %gfut_a[%idx] : !gfut_a_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_a : !gfut_a_buf
  }

  // Issue B global loads at preshuffle byte offsets.
  func.func private @k_load_b_direct(%n_t: index, %k_t: index,
      %B_ptr: !aster_utils.any, %k: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf

    %lid = func.call @lane_id() : () -> index

    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %n_block = affine.apply affine_map<(nt)[nb] -> (nb + nt)>(%i)[%n_base]
        %k_block = affine.apply affine_map<(kt)[kb] -> (kb + kt)>(%kt)[%k]

        %flat_coord = affine.linearize_index [%n_block, %k_block, %lid]
            by ({{N_BLOCKS}}, {{K_BLOCKS}}, 64) : index

        %b_byte_off = layout.linearize %flat_coord,
            #layout.strided_layout<[{{N_BLOCKS}}, {{K_BLOCKS}}, 64] : [{{STRIDE_N0_BYTES}}, 1024, 16]>

        %idx = affine.linearize_index [%kt, %i] by (%k_t, %n_t) : index
        %fut = func.call @load_global_at_byte_off(%B_ptr, %b_byte_off)
            {sched.stage = {{B_STAGE_LOAD}} : i32}
            : (!aster_utils.any, index) -> !future_global_read
        memref.store %fut, %gfut_b[%idx] : !gfut_b_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return %gfut_b : !gfut_b_buf
  }

  // Wait + split + MFMA for both A and B direct. Zero LDS.
  // A wait at A_STAGE_WRITE (same timing as where A LDS write would be).
  // B wait at B_STAGE_WAIT.
  // MFMA at B_A_STAGE_COMPUTE.
  func.func private @k_wait_split_compute_direct_ab(
      %m_t: index, %n_t: index, %k_t: index, %k_mfma: index,
      %gfut_a: !gfut_a_buf, %gfut_b: !gfut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %ub = affine.apply affine_map<()[m, k, n] -> (m * k * n)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %km, %nt = affine.delinearize_index %idx into (%m_t, %k_mfma, %n_t)
          : index, index, index

      %kt = affine.apply affine_map<(km) -> (km floordiv 2)>(%km)
      %kh = affine.apply affine_map<(km) -> (km mod 2)>(%km)

      // Wait A global load at A_STAGE_WRITE.
      %gfut_a_idx = affine.linearize_index [%kt, %mt] by (%k_t, %m_t) : index
      %gfut_a_val = memref.load %gfut_a[%gfut_a_idx] : !gfut_a_buf
      %a_vx4 = func.call @get_global_load_value_vx4(%gfut_a_val)
          {sched.stage = {{A_STAGE_WRITE}} : i32}
          : (!future_global_read) -> !vx4
      %a0, %a1, %a2, %a3 = amdgcn.split_register_range %a_vx4 : !vx4
      %a_k0 = amdgcn.make_register_range %a0, %a1 : !v, !v
      %a_k1 = amdgcn.make_register_range %a2, %a3 : !v, !v
      %a_buf = memref.alloca(%c2) : memref<?x!rt_A_f16>
      memref.store %a_k0, %a_buf[%c0] : memref<?x!rt_A_f16>
      memref.store %a_k1, %a_buf[%c1] : memref<?x!rt_A_f16>
      %a = memref.load %a_buf[%kh] : memref<?x!rt_A_f16>

      // Wait B global load at B_STAGE_WAIT.
      %gfut_b_idx = affine.linearize_index [%kt, %nt] by (%k_t, %n_t) : index
      %gfut_b_val = memref.load %gfut_b[%gfut_b_idx] : !gfut_b_buf
      %b_vx4 = func.call @get_global_load_value_vx4(%gfut_b_val)
          {sched.stage = {{B_STAGE_WAIT}} : i32}
          : (!future_global_read) -> !vx4
      %b0, %b1, %b2, %b3 = amdgcn.split_register_range %b_vx4 : !vx4
      %b_k0 = amdgcn.make_register_range %b0, %b1 : !v, !v
      %b_k1 = amdgcn.make_register_range %b2, %b3 : !v, !v
      %b_buf = memref.alloca(%c2) : memref<?x!rt_B_f16>
      memref.store %b_k0, %b_buf[%c0] : memref<?x!rt_B_f16>
      memref.store %b_k1, %b_buf[%c1] : memref<?x!rt_B_f16>
      %b = memref.load %b_buf[%kh] : memref<?x!rt_B_f16>

      // MFMA at B_A_STAGE_COMPUTE.
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{B_A_STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }
