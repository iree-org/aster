  // Multi-tile copy functions for AMDGCN kernels.
// These functions handle conditional multi-tile global loads and LDS writes
// using the simpler 16x16 wave-level primitives.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/simple-copies.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>

// A 2-level 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions of the outer tile (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - mm_pos, nn_pos: row and column positions of the inner tile (in elements)
//   - elt_size: element size in bytes
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

amdgcn.library @multi_tile_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // copies.mlir
  func.func private @simple_global_load_wave_16x16xf16_wait(!sx2, index, index, index) -> !vx2
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, index, index, index, index)
  func.func private @simple_lds_read_wave_16x16xf16_wait(index, index, index, index) -> !vx2
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2)

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile LDS write
  //===--------------------------------------------------------------------===//
  // Simplified multi-tile LDS write using simple_lds_write_wave_16x16xf16_wait.
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %k: outer loop index (for reading variadic results from load_memref -> mem2reg)
  //   %ii, %jj: tile indices being iterated
  //   %cond_iter: condition index (execute only when cond_iter == 0)
  //   %K, %II, %JJ: loop bounds (unused but kept for API compatibility)
  //   %NT_I, %NT_J: multi-tile factors (write NT_I x NT_J tiles at once)
  //   %lds_base_off: base offset in LDS
  //   %SIZE_J: stride in elements (converted to bytes internally)
  //   %load_memref: input memref[K, NT_I * NT_J] for reading variadic values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @simple_maybe_lds_write_multi_tile
  func.func private @simple_maybe_lds_write_multi_tile(
    %k: index, %ii: index, %jj: index, %cond_iter: index,
    %K: index, %II: index, %JJ: index,
    %NT_I: index, %NT_J: index,
    %lds_base_off: index,
    %SIZE_J: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, NT_I] -> (ii mod NT_I)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, NT_J] -> (jj mod NT_J)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_write {
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      %elt_size = arith.constant 2 : index
      %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_J, elt_size] ->
        (SIZE_J * elt_size)>()[%SIZE_J, %elt_size]

      // Write NT_I x NT_J tiles using simple 16x16 writes
      scf.for %i = %c0 to %NT_I step %c1 {
        scf.for %j = %c0 to %NT_J step %c1 {
          %tile_idx = affine.apply affine_map<()[i, j, NT_J] -> (i * NT_J + j)>()[%i, %j, %NT_J]
          %value = memref.load %load_memref[%k, %tile_idx] : memref<?x?x!vx2>

          %m_pos = affine.apply affine_map<()[ii_pos, i] -> (ii_pos + i * 16)>()[%ii_pos, %i]
          %n_pos = affine.apply affine_map<()[jj_pos, j] -> (jj_pos + j * 16)>()[%jj_pos, %j]

          func.call @simple_lds_write_wave_16x16xf16_wait(
            %value, %lds_base_off, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES)
            : (!vx2, index, index, index, index) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    }
    return
  }
}
