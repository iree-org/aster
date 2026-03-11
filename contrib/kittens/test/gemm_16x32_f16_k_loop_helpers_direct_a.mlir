  // Direct-A helper: A bypasses LDS entirely via ds_bpermute_b32.
  // Appended to base k_loop helpers when a_path="direct".

  // From global_direct_a_16x64_b.mlir
  func.func private @permute_A_fragment(!vx4, index) -> !vx2

  // Extract A values directly from global load futures via ds_bpermute.
  // For each global load future (one per 16x32 tile), waits for the load,
  // then calls permute_A_fragment twice for 2 MFMA K-steps.
  //
  // Output buffer layout: flat [k_mfma_total * m_t] indexed as [k_t, 2, m_t],
  // matching @k_extract_lds_values_a so @k_compute_mfmas works unchanged.
  //
  // TODO: some type safety mechanism to ensure the buffer is [k_t, 2, m_t] and
  // minimize misindexing risks at a distance.
  func.func private @k_extract_direct_a_values(%m_t: index, %k_t: index,
      %gfut_a: !gfut_a_buf) -> !vals_a_buf {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %k_mfma_total = affine.apply affine_map<()[kt] -> (kt * 2)>()[%k_t]
    %buf_size = affine.apply affine_map<()[k, m] -> (k * m)>()[%k_mfma_total, %m_t]
    %a_vals = memref.alloca(%buf_size) : !vals_a_buf
    scf.for %kt = %c0 to %k_t step %c1 {
      scf.for %i = %c0 to %m_t step %c1 {
        %gfut_idx = affine.linearize_index [%kt, %i] by (%k_t, %m_t) : index
        %gfut = memref.load %gfut_a[%gfut_idx] : !gfut_a_buf

        %vx4_data = func.call @get_global_load_value_vx4(%gfut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_global_read) -> !vx4

        // Each 16x32 tile -> 2 MFMA K-steps (K0, K1).
        scf.for %kh = %c0 to %c2 step %c1 {
          %a_frag = func.call @permute_A_fragment(%vx4_data, %kh)
              {sched.stage = {{STAGE_COMPUTE}} : i32}
              : (!vx4, index) -> !vx2
          %flat_idx = affine.linearize_index [%kt, %kh, %i] by (%k_t, %c2, %m_t) : index
          memref.store %a_frag, %a_vals[%flat_idx] : !vals_a_buf
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}
    return %a_vals : !vals_a_buf
  }
