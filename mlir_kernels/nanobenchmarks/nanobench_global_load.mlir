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

  amdgcn.kernel @nanobench_global_load arguments <[
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
    %NT_J = arith.constant {{NUM_TILES}} : index

    // Tile size (in units of f16 elements)
    %TILE_SIZE = arith.constant 16 : index

    // Tile reuse factor
    %TILE_REUSE_FACTOR = arith.constant {{TILE_REUSE_FACTOR}} : index

    // Number of outer iterations, fit all data in L1
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    // DWORDX selector: 1=dword, 2=dwordx2, 3=dwordx3, 4=dwordx4
    %DWORDX = arith.constant {{DWORDX}} : index

    scf.index_switch %DWORDX
    case 1 {
      //===--------------------------------------------------------------------===//
      // dword
      //===--------------------------------------------------------------------===//
      // Allocate memref to hold all tile results
      %memref_vx1 = memref.alloca(%TILE_REUSE_FACTOR, %NT_J) : memref<?x?x!vx1>

      // Outer timing loop
      scf.for %iter = %c0 to %NUM_ITERS step %c1 {
        // Load all tiles into memref
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c1 {
            %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
            %result_vx1 = func.call @global_load_wave_128xf16_via_dword_nowait(
              %ptr,         // global pointer
              %c0, %c0,     // m_pos, n_pos (major tile = 0)
              %c0,          // stride in bytes (single row, stride must not matter)
              %c0, %nn_pos, // mm_pos, nn_pos (minor tile positions)
              %c1           // num_rows
            ) : (!sx2, index, index, index, index, index, index) -> !vx1
            memref.store %result_vx1, %memref_vx1[%reuse, %nt] : memref<?x?x!vx1>
          } {aster.constexpr}
        } {aster.constexpr}

        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>

        // Prevent DCE by reading from memref - erased just before translation to assembly
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c1 {
            %loaded_vx1 = memref.load %memref_vx1[%reuse, %nt] : memref<?x?x!vx1>
            amdgcn.test_inst ins %loaded_vx1 : (!vx1) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
      scf.yield
    }
    case 2 {
      //===--------------------------------------------------------------------===//
      // dwordx2
      //===--------------------------------------------------------------------===//
      // Allocate memref to hold all tile results
      %memref_vx2 = memref.alloca(%TILE_REUSE_FACTOR, %NT_J) : memref<?x?x!vx2>

      // Outer timing loop
      scf.for %iter = %c0 to %NUM_ITERS step %c1 {
        // Load all tiles into memref
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c2 {
            %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
            %result_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_nowait(
              %ptr,         // global pointer
              %c0, %c0,     // m_pos, n_pos (major tile = 0)
              %c0,          // stride in bytes (single row, stride must not matter)
              %c0, %nn_pos, // mm_pos, nn_pos (minor tile positions)
              %c1           // num_rows
            ) : (!sx2, index, index, index, index, index, index) -> !vx2
            memref.store %result_vx2, %memref_vx2[%reuse, %nt] : memref<?x?x!vx2>
          } {aster.constexpr}
        } {aster.constexpr}

        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>

        // Prevent DCE by reading from memref - erased just before translation to assembly
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c2 {
            %loaded_vx2 = memref.load %memref_vx2[%reuse, %nt] : memref<?x?x!vx2>
            amdgcn.test_inst ins %loaded_vx2 : (!vx2) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
      scf.yield
    }
    case 3 {
      //===--------------------------------------------------------------------===//
      // dwordx3
      //===--------------------------------------------------------------------===//
      // Allocate memref to hold all tile results
      %memref_vx3 = memref.alloca(%TILE_REUSE_FACTOR, %NT_J) : memref<?x?x!vx3>

      // Outer timing loop
      scf.for %iter = %c0 to %NUM_ITERS step %c1 {
        // Load all tiles into memref
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c3 {
            %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
            %result_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_nowait(
              %ptr,         // global pointer
              %c0, %c0,     // m_pos, n_pos (major tile = 0)
              %c0,          // stride in bytes (single row, stride must not matter)
              %c0, %nn_pos, // mm_pos, nn_pos (minor tile positions)
              %c1           // num_rows
            ) : (!sx2, index, index, index, index, index, index) -> !vx3
            memref.store %result_vx3, %memref_vx3[%reuse, %nt] : memref<?x?x!vx3>
          } {aster.constexpr}
        } {aster.constexpr}

        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>

        // Prevent DCE by reading from memref - erased just before translation to assembly
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c3 {
            %loaded_vx3 = memref.load %memref_vx3[%reuse, %nt] : memref<?x?x!vx3>
            amdgcn.test_inst ins %loaded_vx3 : (!vx3) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
      scf.yield
    }
    case 4 {
      //===--------------------------------------------------------------------===//
      // dwordx4
      //===--------------------------------------------------------------------===//
      // Allocate memref to hold all tile results
      %memref_vx4 = memref.alloca(%TILE_REUSE_FACTOR, %NT_J) : memref<?x?x!vx4>

      // Outer timing loop
      scf.for %iter = %c0 to %NUM_ITERS step %c1 {
        // Load all tiles into memref
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c4 {
            %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
            %result_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_nowait(
              %ptr,         // global pointer
              %c0, %c0,     // m_pos, n_pos (major tile = 0)
              %c0,          // stride in bytes (single row, stride must not matter)
              %c0, %nn_pos, // mm_pos, nn_pos (minor tile positions)
              %c1           // num_rows
            ) : (!sx2, index, index, index, index, index, index) -> !vx4
            memref.store %result_vx4, %memref_vx4[%reuse, %nt] : memref<?x?x!vx4>
          } {aster.constexpr}
        } {aster.constexpr}

        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>

        // Prevent DCE by reading from memref - erased just before translation to assembly
        scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
          scf.for %nt = %c0 to %NT_J step %c4 {
            %loaded_vx4 = memref.load %memref_vx4[%reuse, %nt] : memref<?x?x!vx4>
            amdgcn.test_inst ins %loaded_vx4 : (!vx4) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
      scf.yield
    }
    default {
      amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 2
    }

    amdgcn.end_kernel
  }
}
