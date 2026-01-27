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

// A 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

// A 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - m_pos, n_pos: row and column positions (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2-level 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - mm_pos, nn_pos: row and column positions of the minor tile (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2-level 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions of the outer tile (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - mm_pos, nn_pos: row and column positions of the inner tile (in elements)
//   - elt_size: element size in bytes
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

amdgcn.library @multi_tile_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // simple-copies.mlir
  func.func private @simple_global_load_wave_16x16xf16_wait(!tensor_position_descriptor_2d) -> !vx2
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, !lds_position_descriptor_2d)
  func.func private @simple_lds_read_wave_16x16xf16_wait(!lds_position_descriptor_2d) -> !vx2
  // copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2)

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile LDS write
  //===--------------------------------------------------------------------===//
  // Simplified multi-tile LDS write using simple_lds_write_wave_16x16xf16_wait.
  // Executes when cond_iter == 0 AND m_pos % NT_I == 0 AND n_pos % NT_J == 0.
  //
  // Parameters:
  //   %k: outer loop index (for reading variadic results from load_memref -> mem2reg)
  //   %cond_iter: condition index (execute only when cond_iter == 0)
  //   %K, %II, %JJ: loop bounds (unused but kept for API compatibility)
  //   %NT_I, %NT_J: multi-tile factors (write NT_I x NT_J tiles at once)
  //   %lds_pos_desc_base: LDS position descriptor (m_pos/n_pos are tile indices, converted to elements internally)
  //   %load_memref: input memref[K, NT_I * NT_J] for reading variadic values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @simple_maybe_lds_write_multi_tile
  func.func private @simple_maybe_lds_write_multi_tile(
    %k: index, %cond_iter: index,
    %K: index, %II: index, %JJ: index,
    %NT_I: index, %NT_J: index,
    %lds_pos_desc_base: !lds_position_descriptor_2d,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract tile indices from descriptor (m_pos/n_pos are tile indices here)
    %ii = aster_utils.struct_extract %lds_pos_desc_base["m_pos"] : !lds_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %lds_pos_desc_base["n_pos"] : !lds_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, NT_I] -> (ii mod NT_I)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, NT_J] -> (jj mod NT_J)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_write {
      // Extract remaining fields from base descriptor
      %lds_base = aster_utils.struct_extract %lds_pos_desc_base["lds_base"] : !lds_position_descriptor_2d -> index
      %lds_stride_in_bytes = aster_utils.struct_extract %lds_pos_desc_base["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %lds_pos_desc_base["elt_size"] : !lds_position_descriptor_2d -> index

      // Write NT_I x NT_J tiles using simple 16x16 writes
      scf.for %i = %c0 to %NT_I step %c1 {
        scf.for %j = %c0 to %NT_J step %c1 {
          %tile_idx = affine.apply affine_map<()[i, j, NT_J] -> (i * NT_J + j)>()[%i, %j, %NT_J]
          %value = memref.load %load_memref[%k, %tile_idx] : memref<?x?x!vx2>

          // Convert tile indices to element positions: (ii + i) * 16, (jj + j) * 16
          %m_pos = affine.apply affine_map<()[ii, i] -> ((ii + i) * 16)>()[%ii, %i]
          %n_pos = affine.apply affine_map<()[jj, j] -> ((jj + j) * 16)>()[%jj, %j]

          %lds_pos_desc = aster_utils.struct_create(%lds_base, %m_pos, %n_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          func.call @simple_lds_write_wave_16x16xf16_wait(%value, %lds_pos_desc)
            : (!vx2, !lds_position_descriptor_2d) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    }
    return
  }
}
