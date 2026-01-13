// Nanobenchmark for @global_load_wave_256xf16_via_dwordx2_wait
// Measures global load performance with data fitting in L1 cache.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library declaration
  // %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %num_rows
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !sx2, index, index, index, index, index, index) -> !vx2

  amdgcn.kernel @nanobench_global_load_multi_tile arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {shared_memory_size = 0 : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %ptr_s = amdgcn.load_arg 0 : !sx2
    %ptr = lsir.assume_noalias %ptr_s : (!sx2) -> !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters for 4-tile configuration (1x4 tiles = 16x64 region)
    %NT_I = arith.constant 1 : index      // 1 tile in M
    %NT_J = arith.constant 4 : index      // 4 tiles in N

    // Derived parameters via affine.apply
    %row_size = affine.apply affine_map<()[n] -> (16 ceildiv n)>()[%NT_J]
    %col_size = affine.apply affine_map<()[n] -> (16 * n)>()[%NT_J]
    %total_rows = affine.apply affine_map<()[m] -> (16 * m)>()[%NT_I]
    %total_cols = affine.apply affine_map<()[n] -> (16 * n)>()[%NT_J]
    %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[n] -> (16 * n * 2)>()[%NT_J]

    // Number of outer iterations, fit all data in L1
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Load NT_I * NT_J fragments covering total_rows x total_cols region
      scf.for %mt = %c0 to %total_rows step %row_size {
        scf.for %nt = %c0 to %total_cols step %col_size {
          %result = func.call @global_load_wave_256xf16_via_dwordx2_wait(
            %ptr,                    // global pointer
            %c0, %c0,                // m_pos, n_pos (major tile = 0)
            %GLOBAL_STRIDE_IN_BYTES, // stride in bytes
            %mt, %nt,                // mm_pos, nn_pos (minor tile positions)
            %row_size                // num_rows
          ) : (!sx2, index, index, index, index, index, index) -> !vx2

          // Prevent DCE - erased just before translation to assembly
          amdgcn.test_inst ins %result : (!vx2) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
