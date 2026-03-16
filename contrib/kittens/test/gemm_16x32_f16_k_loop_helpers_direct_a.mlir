  // Direct-A helper: A bypasses LDS entirely via ds_bpermute_b32.
  // Appended to base k_loop helpers when a_path="direct".

  // From global_direct_a_16x64_b.mlir (split issue/resolve)
  func.func private @issue_A_bpermutes(!vx4, index) -> !future_lds_read
  func.func private @resolve_A_fragment(!future_lds_read) -> !vx2

  // Issue all A bpermute permutations, returning futures.
  // NO waits happen here -- tokens flow to the top-level k_loop.
  //
  // Output buffer layout: flat [k_mfma_total * m_t] indexed as [k_t, 2, m_t],
  // matching @k_read_lds_at_addrs_a so @k_wait_and_compute_mfmas works unchanged.
  func.func private @k_extract_direct_a_futures(%m_t: index, %k_t: index,
      %gfut_a: !gfut_a_buf) -> !fut_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[k, m] -> (k * m)>()[%k_mfma_total, %m_t]

    %a_futs = memref.alloca(%buf_size) : !fut_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gfut = memref.load %gfut_a[%gfut_idx] : !gfut_a_buf

        %vx4_data = func.call @get_global_load_value_vx4(%gfut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_global_read) -> !vx4

        // Each 16x32 tile -> 2 MFMA K-steps (K0, K1).
        // Issue bpermutes only -- no waits.
        scf.for %kh = %c0 to %c2 step %c1 {
          %a_fut = func.call @issue_A_bpermutes(%vx4_data, %kh)
              {sched.stage = {{STAGE_COMPUTE}} : i32}
              : (!vx4, index) -> !future_lds_read
          %flat_idx = affine.linearize_index [%kt, %kh, %i] by (%k_t, %c2, %m_t) : index
          memref.store %a_fut, %a_futs[%flat_idx] : !fut_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    return %a_futs : !fut_a_buf
  }

  // Wait + compute MFMAs for direct-A path.
  // A futures hold bpermuted vx4 (not vx2) -- resolved via @resolve_A_fragment.
  // B futures are standard LDS reads resolved via @get_lds_read_value_vx2.
  // Resolves each future right before the consuming MFMA for maximum
  // latency hiding between bpermute issue and data consumption.
  func.func private @k_wait_and_compute_mfmas_direct_a(
      %m_t: index, %n_t: index, %k_mfma: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ub = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%m_t, %k_mfma, %n_t]
    scf.for %idx = %c0 to %ub step %c1 {
      %mt, %km, %nt = affine.delinearize_index %idx into (%m_t, %k_mfma, %n_t) : index, index, index
      %a_idx = affine.linearize_index [%km, %mt] by (%k_mfma, %m_t) : index
      %b_idx = affine.linearize_index [%km, %nt] by (%k_mfma, %n_t) : index

      // A: resolve bpermute future (wait + cndmask -> vx2)
      %fut_a = memref.load %a_fut[%a_idx] : !fut_a_buf
      %a = func.call @resolve_A_fragment(%fut_a)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16

      // B: standard LDS resolve
      %fut_b = memref.load %b_fut[%b_idx] : !fut_b_buf
      %b = func.call @get_lds_read_value_vx2(%fut_b)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16

      // MFMA
      %c_idx = affine.linearize_index [%mt, %nt] by (%m_t, %n_t) : index
      %c_old = memref.load %c_buf[%c_idx] : !c_buf
      %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
      memref.store %c_new, %c_buf[%c_idx] : !c_buf
    } {aster.constexpr}
    return
  }
