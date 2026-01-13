// Nanobenchmark for @global_load_wave_xxx_wait
// Measures global load performance with data fitting in L1 cache.

!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr_range<[? + 1]>
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx3 = !amdgcn.sgpr_range<[? + 3]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // copies.mlir
  func.func private @global_load_wave_128xf16_via_dword_wait(
    !sx2, index, index, index, index, index, index) -> !vx1
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !sx2, index, index, index, index, index, index) -> !vx2
  func.func private @global_load_wave_384xf16_via_dwordx3_wait(
    !sx2, index, index, index, index, index, index) -> !vx3
  func.func private @global_load_wave_512xf16_via_dwordx4_wait(
    !sx2, index, index, index, index, index, index) -> !vx4

  amdgcn.kernel @nanobench_global_load_multi_tile arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {shared_memory_size = 0 : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %ptr_s = amdgcn.load_arg 0 : !sx2
    %ptr = lsir.assume_noalias %ptr_s : (!sx2) -> !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters for 1x4 tile configuration
    %NT_I = arith.constant 1 : index // 1 tile in M
    %NT_J = arith.constant 4 : index // 4 tiles in N

    // Number of outer iterations, fit all data in L1
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    //===--------------------------------------------------------------------===//
    // dword
    //===--------------------------------------------------------------------===//
    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      scf.for %mt = %c0 to %c1 step %c1 {
        scf.for %nt = %c0 to %NT_J step %c1 {
          %result_vx1 = func.call @global_load_wave_128xf16_via_dword_wait(
            %ptr,     // global pointer
            %c0, %c0, // m_pos, n_pos (major tile = 0)
            %c0,      // stride in bytes (single row, stride must not matter)
            %mt, %nt, // mm_pos, nn_pos (minor tile positions)
            %c1       // num_rows
          ) : (!sx2, index, index, index, index, index, index) -> !vx1

          // Prevent DCE - erased just before translation to assembly
          amdgcn.test_inst ins %result_vx1 : (!vx1) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.sopp.sopp <s_barrier>

    //===--------------------------------------------------------------------===//
    // dwordx2
    //===--------------------------------------------------------------------===//
    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      scf.for %mt = %c0 to %c1 step %c1 {
        scf.for %nt = %c0 to %NT_J step %c1 {
          %result_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_wait(
            %ptr,     // global pointer
            %c0, %c0, // m_pos, n_pos (major tile = 0)
            %c0,      // stride in bytes (single row, stride must not matter)
            %mt, %nt, // mm_pos, nn_pos (minor tile positions)
            %c1       // num_rows
          ) : (!sx2, index, index, index, index, index, index) -> !vx2

          // Prevent DCE - erased just before translation to assembly
          amdgcn.test_inst ins %result_vx2 : (!vx2) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.sopp.sopp <s_barrier>

    //===--------------------------------------------------------------------===//
    // dwordx3
    //===--------------------------------------------------------------------===//
    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      scf.for %mt = %c0 to %c1 step %c1 {
        scf.for %nt = %c0 to %NT_J step %c1 {
          %result_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_wait(
            %ptr,     // global pointer
            %c0, %c0, // m_pos, n_pos (major tile = 0)
            %c0,      // stride in bytes (single row, stride must not matter)
            %mt, %nt, // mm_pos, nn_pos (minor tile positions)
            %c1       // num_rows
          ) : (!sx2, index, index, index, index, index, index) -> !vx3

          // Prevent DCE - erased just before translation to assembly
          amdgcn.test_inst ins %result_vx3 : (!vx3) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.sopp.sopp <s_barrier>

    //===--------------------------------------------------------------------===//
    // dwordx4
    //===--------------------------------------------------------------------===//
    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      scf.for %mt = %c0 to %c1 step %c1 {
        scf.for %nt = %c0 to %NT_J step %c1 {
          %result_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_wait(
            %ptr,     // global pointer
            %c0, %c0, // m_pos, n_pos (major tile = 0)
            %c0,      // stride in bytes (single row, stride must not matter)
            %mt, %nt, // mm_pos, nn_pos (minor tile positions)
            %c1       // num_rows
          ) : (!sx2, index, index, index, index, index, index) -> !vx4

          // Prevent DCE - erased just before translation to assembly
          amdgcn.test_inst ins %result_vx4 : (!vx4) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.sopp.sopp <s_barrier>

    amdgcn.end_kernel
  }
}
