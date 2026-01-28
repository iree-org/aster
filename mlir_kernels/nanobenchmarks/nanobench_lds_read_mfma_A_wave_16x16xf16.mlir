// Nanobenchmark for @lds_read_A_wave_16x16xf16_fragment_wait
// Uses garbage values in LDS (no verification).

// From descriptors.mlir
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library declarations
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    !lds_position_descriptor_2d, i1) -> !vx2

  amdgcn.kernel @nanobench_lds_read_mfma_A_wave_16x16xf16
  attributes {shared_memory_size = {{LDS_SIZE}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters matching typical GEMM usage
    %II = arith.constant 4 : index       // Number of tiles in M dimension
    %JJ = arith.constant 8 : index       // Number of tiles in K dimension

    %elt_size = arith.constant 2 : index
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[JJ, elt_size]
      -> (JJ * 16 * elt_size)>()[%JJ, %elt_size]

    // Number of outer iterations
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    // Outer timing loop - repeat the tile iterations NUM_ITERS times
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Inner loop over all tile indices
      scf.for %ii = %c0 to %II step %c1 {
        scf.for %jj = %c0 to %JJ step %c1 {
          // m_pos and n_pos are tile indices * 16
          %m_pos = arith.muli %ii, %c1 : index  // Would be ii * 16 in real usage
          %n_pos = arith.muli %jj, %c1 : index  // Would be jj * 16 in real usage

          // Call the LDS read function
          %false = arith.constant false
          %lds_pos_desc = aster_utils.struct_create(%c0, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          %result = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc, %false)
            : (!lds_position_descriptor_2d, i1) -> !vx2
          // Prevent DCE - erased just before translation to assembly with amdgcn-remove-test-inst
          amdgcn.test_inst ins %result : (!vx2) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }
}
