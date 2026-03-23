  // Direct-B helpers: B bypasses LDS via MFMA-aware preshuffle.
  // Pipeline: B load at A_STAGE_LOAD, B wait+split at A_STAGE_WRITE,
  //           MFMA at A_STAGE_COMPUTE.
  //
  // The preshuffle byte offset is computed via layout.linearize with a
  // static layout attribute matching the Python-side preshuffle_b_index_layout.
  // This is the single source of truth for the B addressing.

  func.func private @lane_id() -> index

  // Issue B global loads at preshuffle byte offsets (A_STAGE_LOAD).
  // The preshuffle layout maps (n_block, k_block, lane_id) -> byte offset.
  // Byte strides: (STRIDE_N0_BYTES, 1024, 16) where STRIDE_N0_BYTES = K * 32.
  func.func private @k_load_b_direct(%n_t: index, %k_t: index,
      %B_ptr: !aster_utils.any, %k: index, %n_base: index)
      -> !gfut_b_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buf_size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%k_t, %n_t]
    %gfut_b = memref.alloca(%buf_size) : !gfut_b_buf

    // Compute the flat coordinate for layout.linearize: n_block * (K_BLOCKS * 64) + k_block * 64 + lane_id.
    %lid = func.call @lane_id() : () -> index

    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %n_t step %c1 {
        %n_block = affine.apply affine_map<(nt)[nb] -> (nb + nt)>(%i)[%n_base]
        %k_block = affine.apply affine_map<(kt)[kb] -> (kb + kt)>(%kt)[%k]

        // Flat coordinate into preshuffle layout: (n_block, k_block, lane_id)
        // linearized by sizes (N_BLOCKS, K_BLOCKS, 64).
        %flat_coord = affine.linearize_index [%n_block, %k_block, %lid]
            by ({{N_BLOCKS}}, {{K_BLOCKS}}, 64) : index

        // Preshuffle byte offset via layout.linearize.
        // Layout: [N_BLOCKS, K_BLOCKS, 64] : [STRIDE_N0_BYTES, 1024, 16]
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

  // Wait B loads (A_STAGE_WRITE) + split + MFMA (A_STAGE_COMPUTE).
  func.func private @k_wait_split_compute_direct_b(
      %m_t: index, %n_t: index, %k_t: index, %k_mfma: index,
      %gfut_b: !gfut_b_buf, %a_fut: !fut_a_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %ub = affine.apply affine_map<()[m, k, n] -> (m * k * n)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %km, %nt = affine.delinearize_index %idx into (%m_t, %k_mfma, %n_t)
          : index, index, index

      %kt = affine.apply affine_map<(km) -> (km floordiv 2)>(%km)
      %kh = affine.apply affine_map<(km) -> (km mod 2)>(%km)

      // Wait B global load at A_STAGE_WRITE.
      %gfut_idx = affine.linearize_index [%kt, %nt] by (%k_t, %n_t) : index
      %gfut = memref.load %gfut_b[%gfut_idx] : !gfut_b_buf
      %vx4_data = func.call @get_global_load_value_vx4(%gfut)
          {sched.stage = {{B_STAGE_WAIT}} : i32}
          : (!future_global_read) -> !vx4

      // Split vx4 (stage inherited via propagation).
      %v0, %v1, %v2, %v3 = amdgcn.split_register_range %vx4_data : !vx4
      %b_k0 = amdgcn.make_register_range %v0, %v1 : !v, !v
      %b_k1 = amdgcn.make_register_range %v2, %v3 : !v, !v

      // Select K-half (constexpr, eliminated after expansion + SROA).
      %b_buf = memref.alloca(%c2) : memref<?x!rt_B_f16>
      memref.store %b_k0, %b_buf[%c0] : memref<?x!rt_B_f16>
      memref.store %b_k1, %b_buf[%c1] : memref<?x!rt_B_f16>
      %b = memref.load %b_buf[%kh] : memref<?x!rt_B_f16>

      // A from LDS at A_STAGE_COMPUTE.
      %a_idx = affine.linearize_index [%km, %mt] by (%k_mfma, %m_t) : index
      %fut_a = memref.load %a_fut[%a_idx] : !fut_a_buf
      %a = func.call @get_lds_read_value_vx2(%fut_a)
          {sched.stage = {{B_A_STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16

      // MFMA at A_STAGE_COMPUTE.
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{B_A_STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }
