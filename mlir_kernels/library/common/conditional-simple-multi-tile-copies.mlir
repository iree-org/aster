// Conditional simple multi-tile copy functions for AMDGCN kernels.
//
// Provides conditional multi-tile primitives using the simple 16x16 wave-level
// primitives from simple-copies.mlir (non-coalesced, fixed num_rows=16).
// For coalesced variants, use conditional-multi-tile-copies.mlir.
//
// Naming convention: @maybe_simple_<operation>_wave_multi_tile_16x16xf16

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
!conditional_execution_descriptor_2d = !aster_utils.struct<cond_zero: index, cond_mod_zero_i: index, cond_mod_zero_j: index>

//===-----------------------------------------------------------------------===//
// Wave-level, conditional simple multi-tile global load instructions,
// parameterizable by !conditional_execution_descriptor_2d and !tensor_position_descriptor_2d.
//
// Conditionally loads cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles using simple (non-coalesced) loads.
// Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_simple_multi_tile_global_load_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From simple-copies.mlir
  func.func private @simple_global_load_wave_16x16xf16_wait(!tensor_position_descriptor_2d) -> !vx2

  //===--------------------------------------------------------------------===//
  // Conditional simple multi-tile global load
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via simple_global_load (non-coalesced)
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally loads cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from global memory.
  // Uses @simple_global_load_wave_16x16xf16_wait for each tile (non-coalesced).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - cond_zero: execute only when == 0
  //     - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors
  //   %tensor_desc: !tensor_position_descriptor_2d
  //     - ptr: base pointer
  //     - m_pos, n_pos: tile indices (converted to m*16, n*16 internally)
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %k: loop index for memref storage
  //   %load_memref: memref<?x?x!vx2> - output memref[K, cond_mod_zero_i * cond_mod_zero_j]

  // CHECK-LABEL: func.func private @maybe_simple_global_load_wave_multi_tile_16x16xf16
  func.func private @maybe_simple_global_load_wave_multi_tile_16x16xf16(
    %cond_desc: !conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2d,
    %k: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_i = aster_utils.struct_extract %cond_desc["cond_mod_zero_i"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_j = aster_utils.struct_extract %cond_desc["cond_mod_zero_j"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (m_pos/n_pos are tile indices here)
    %ii = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, cond_mod_zero_i] -> (ii mod cond_mod_zero_i)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, cond_mod_zero_j] -> (jj mod cond_mod_zero_j)>()[%jj, %cond_mod_zero_j]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_load {
      // Extract remaining fields from descriptor
      %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2d -> !sx2
      %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2d -> index

      // Load cond_mod_zero_i x cond_mod_zero_j tiles using simple 16x16 loads
      scf.for %i = %c0 to %cond_mod_zero_i step %c1 {
        scf.for %j = %c0 to %cond_mod_zero_j step %c1 {
          // Convert tile indices to element positions: (ii + i) * 16, (jj + j) * 16
          %m_pos = affine.apply affine_map<()[ii, i] -> ((ii + i) * 16)>()[%ii, %i]
          %n_pos = affine.apply affine_map<()[jj, j] -> ((jj + j) * 16)>()[%jj, %j]

          %pos_desc = aster_utils.struct_create(%ptr, %m_pos, %n_pos, %global_stride_in_bytes, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
          %value = func.call @simple_global_load_wave_16x16xf16_wait(%pos_desc)
            : (!tensor_position_descriptor_2d) -> !vx2

          %tile_idx = affine.apply affine_map<()[i, j, cond_mod_zero_j] -> (i * cond_mod_zero_j + j)>()[%i, %j, %cond_mod_zero_j]
          memref.store %value, %load_memref[%k, %tile_idx] : memref<?x?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    }

    return
  }
}

//===-----------------------------------------------------------------------===//
// Wave-level, conditional simple multi-tile LDS write instructions,
// parameterizable by !conditional_execution_descriptor_2d and !lds_position_descriptor_2d.
//
// Conditionally writes cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles using simple (non-coalesced) writes.
// Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_simple_multi_tile_lds_write_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From simple-copies.mlir
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, !lds_position_descriptor_2d)

  //===--------------------------------------------------------------------===//
  // Conditional simple multi-tile LDS write
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via simple_lds_write (non-coalesced)
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally writes cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from VGPRs to LDS.
  // Uses @simple_lds_write_wave_16x16xf16_wait for each tile (non-coalesced).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - cond_zero: execute only when == 0
  //     - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors
  //   %lds_pos_desc_base: !lds_position_descriptor_2d
  //     - lds_base: base offset in LDS (bytes)
  //     - m_pos, n_pos: tile indices (converted to m*16, n*16 internally)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %k: loop index for memref access
  //   %load_memref: memref<?x?x!vx2> - input memref[K, cond_mod_zero_i * cond_mod_zero_j]

  // CHECK-LABEL: func.func private @maybe_simple_lds_write_wave_multi_tile_16x16xf16
  func.func private @maybe_simple_lds_write_wave_multi_tile_16x16xf16(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_pos_desc_base: !lds_position_descriptor_2d,
    %k: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_i = aster_utils.struct_extract %cond_desc["cond_mod_zero_i"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_j = aster_utils.struct_extract %cond_desc["cond_mod_zero_j"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (m_pos/n_pos are tile indices here)
    %ii = aster_utils.struct_extract %lds_pos_desc_base["m_pos"] : !lds_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %lds_pos_desc_base["n_pos"] : !lds_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, cond_mod_zero_i] -> (ii mod cond_mod_zero_i)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, cond_mod_zero_j] -> (jj mod cond_mod_zero_j)>()[%jj, %cond_mod_zero_j]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_write {
      // Extract remaining fields from base descriptor
      %lds_base = aster_utils.struct_extract %lds_pos_desc_base["lds_base"] : !lds_position_descriptor_2d -> index
      %lds_stride_in_bytes = aster_utils.struct_extract %lds_pos_desc_base["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %lds_pos_desc_base["elt_size"] : !lds_position_descriptor_2d -> index

      // Write cond_mod_zero_i x cond_mod_zero_j tiles using simple 16x16 writes
      scf.for %i = %c0 to %cond_mod_zero_i step %c1 {
        scf.for %j = %c0 to %cond_mod_zero_j step %c1 {
          %tile_idx = affine.apply affine_map<()[i, j, cond_mod_zero_j] -> (i * cond_mod_zero_j + j)>()[%i, %j, %cond_mod_zero_j]
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
