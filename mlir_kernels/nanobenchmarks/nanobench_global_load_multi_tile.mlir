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
  func.func private @global_load_wave_128xf16_via_dword_nowait(
    !sx2, index, index, index, index, index, index) -> !vx1
  func.func private @global_load_wave_256xf16_via_dwordx2_nowait(
    !sx2, index, index, index, index, index, index) -> !vx2
  func.func private @global_load_wave_384xf16_via_dwordx3_nowait(
    !sx2, index, index, index, index, index, index) -> !vx3
  func.func private @global_load_wave_512xf16_via_dwordx4_nowait(
    !sx2, index, index, index, index, index, index) -> !vx4

  amdgcn.kernel @nanobench_global_load_multi_tile arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %ptr_s = amdgcn.load_arg 0 : !sx2
    %ptr = lsir.assume_noalias %ptr_s : (!sx2) -> !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Parameters for {{NUM_TILES}} tile configuration
    %NT_J = arith.constant {{NUM_TILES}} : index // 4 tiles in N

    // Number of outer iterations, fit all data in L1
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    //===--------------------------------------------------------------------===//
    // dword
    //===--------------------------------------------------------------------===//
    // Allocate memref to hold all tile results
    %memref_vx1 = memref.alloca(%NT_J) : memref<?x!vx1>

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Load all tiles into memref
      scf.for %nt = %c0 to %NT_J step %c1 {
        %result_vx1 = func.call @global_load_wave_128xf16_via_dword_nowait(
          %ptr,     // global pointer
          %c0, %c0, // m_pos, n_pos (major tile = 0)
          %c0,      // stride in bytes (single row, stride must not matter)
          %c0, %nt, // mm_pos, nn_pos (minor tile positions)
          %c1       // num_rows
        ) : (!sx2, index, index, index, index, index, index) -> !vx1
        memref.store %result_vx1, %memref_vx1[%nt] : memref<?x!vx1>
      } {aster.constexpr}

      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

      // Prevent DCE by reading from memref - erased just before translation to assembly
      scf.for %nt = %c0 to %NT_J step %c1 {
        %loaded_vx1 = memref.load %memref_vx1[%nt] : memref<?x!vx1>
        amdgcn.test_inst ins %loaded_vx1 : (!vx1) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    //===--------------------------------------------------------------------===//
    // dwordx2
    //===--------------------------------------------------------------------===//
    // Allocate memref to hold all tile results
    %memref_vx2 = memref.alloca(%NT_J) : memref<?x!vx2>

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Load all tiles into memref
      scf.for %nt = %c0 to %NT_J step %c2 {
        %result_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_nowait(
          %ptr,     // global pointer
          %c0, %c0, // m_pos, n_pos (major tile = 0)
          %c0,      // stride in bytes (single row, stride must not matter)
          %c0, %nt, // mm_pos, nn_pos (minor tile positions)
          %c1       // num_rows
        ) : (!sx2, index, index, index, index, index, index) -> !vx2
        memref.store %result_vx2, %memref_vx2[%nt] : memref<?x!vx2>
      } {aster.constexpr}

      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

      // Prevent DCE by reading from memref - erased just before translation to assembly
      scf.for %nt = %c0 to %NT_J step %c2 {
        %loaded_vx1 = memref.load %memref_vx2[%nt] : memref<?x!vx2>
        amdgcn.test_inst ins %loaded_vx1 : (!vx2) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    //===--------------------------------------------------------------------===//
    // dwordx3
    //===--------------------------------------------------------------------===//
    // Allocate memref to hold all tile results
    %memref_vx3 = memref.alloca(%NT_J) : memref<?x!vx3>

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Load all tiles into memref
      scf.for %nt = %c0 to %NT_J step %c3 {
        %result_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_nowait(
          %ptr,     // global pointer
          %c0, %c0, // m_pos, n_pos (major tile = 0)
          %c0,      // stride in bytes (single row, stride must not matter)
          %c0, %nt, // mm_pos, nn_pos (minor tile positions)
          %c1       // num_rows
        ) : (!sx2, index, index, index, index, index, index) -> !vx3
        memref.store %result_vx3, %memref_vx3[%nt] : memref<?x!vx3>
      } {aster.constexpr}

      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

      // Prevent DCE by reading from memref - erased just before translation to assembly
      scf.for %nt = %c0 to %NT_J step %c3 {
        %loaded_vx3 = memref.load %memref_vx3[%nt] : memref<?x!vx3>
        amdgcn.test_inst ins %loaded_vx3 : (!vx3) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    //===--------------------------------------------------------------------===//
    // dwordx4
    //===--------------------------------------------------------------------===//
    // Allocate memref to hold all tile results
    %memref_vx4 = memref.alloca(%NT_J) : memref<?x!vx4>

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Load all tiles into memref
      scf.for %nt = %c0 to %NT_J step %c4 {
        %result_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_nowait(
          %ptr,     // global pointer
          %c0, %c0, // m_pos, n_pos (major tile = 0)
          %c0,      // stride in bytes (single row, stride must not matter)
          %c0, %nt, // mm_pos, nn_pos (minor tile positions)
          %c1       // num_rows
        ) : (!sx2, index, index, index, index, index, index) -> !vx4
        memref.store %result_vx4, %memref_vx4[%nt] : memref<?x!vx4>
      } {aster.constexpr}

      amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

      // Prevent DCE by reading from memref - erased just before translation to assembly
      scf.for %nt = %c0 to %NT_J step %c4 {
        %loaded_vx4 = memref.load %memref_vx4[%nt] : memref<?x!vx4>
        amdgcn.test_inst ins %loaded_vx4 : (!vx4) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.end_kernel
  }
}
