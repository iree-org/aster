// Simple multi-tile copy functions for AMDGCN kernels.
//
// Provides simplified conditional multi-tile LDS write primitives using
// the simple 16x16 wave-level primitives from simple-copies.mlir.
// For coalesced variants, use conditional-multi-tile-copies.mlir.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/simple-copies.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

//===-----------------------------------------------------------------------===//
// Wave-level, conditional simple multi-tile LDS write instructions,
// parameterizable by !conditional_execution_descriptor_2d and !lds_position_descriptor_2d.
//
// Conditionally writes NT_I x NT_J 16x16xf16 tiles using simple (non-coalesced) writes.
// Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_simple_multi_tile_lds_write_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From simple-copies.mlir
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, !lds_position_descriptor_2d)

  //===--------------------------------------------------------------------===//
  // Conditional simple multi-tile LDS write
  //   NT_I x NT_J 16x16xf16 tiles via simple_lds_write (non-coalesced)
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally writes NT_I x NT_J 16x16xf16 tiles from VGPRs to LDS.
  // Uses @simple_lds_write_wave_16x16xf16_wait for each tile (non-coalesced).
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - k: loop index for memref access
  //     - cond_iter: execute only when == 0
  //     - NT_I, NT_J: multi-tile factors
  //   %lds_pos_desc_base: !lds_position_descriptor_2d
  //     - lds_base: base offset in LDS (bytes)
  //     - m_pos, n_pos: tile indices (converted to m*16, n*16 internally)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %load_memref: memref<?x?x!vx2> - input memref[K, NT_I * NT_J]

  // CHECK-LABEL: func.func private @simple_maybe_lds_write_multi_tile
  func.func private @simple_maybe_lds_write_multi_tile(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_pos_desc_base: !lds_position_descriptor_2d,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from conditional execution descriptor
    %k = aster_utils.struct_extract %cond_desc["k"] : !conditional_execution_descriptor_2d -> index
    %cond_iter = aster_utils.struct_extract %cond_desc["cond_iter"] : !conditional_execution_descriptor_2d -> index
    %NT_I = aster_utils.struct_extract %cond_desc["NT_I"] : !conditional_execution_descriptor_2d -> index
    %NT_J = aster_utils.struct_extract %cond_desc["NT_J"] : !conditional_execution_descriptor_2d -> index

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
