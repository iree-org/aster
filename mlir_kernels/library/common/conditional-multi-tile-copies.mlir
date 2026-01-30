// Conditional multi-tile copy functions for AMDGCN kernels.
//
// Provides conditional variants of the multi-tile copy primitives in multi-tile-copies.mlir.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

amdgcn.library @conditional_multi_tile_copies isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // External function declarations
  //===--------------------------------------------------------------------===//

  // From simple-copies.mlir
  func.func private @simple_global_load_wave_16x16xf16_wait(!tensor_position_descriptor_2d) -> !vx2

  // From multi-tile-copies.mlir
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()

  //===--------------------------------------------------------------------===//
  // Simple conditional multi-tile global loads
  //   Uses simple_global_load_wave_16x16xf16_wait (no coalescing optimization)
  //===--------------------------------------------------------------------===//
  // Simplified multi-tile global load using simple_global_load_wave_16x16xf16_wait.
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %cond_desc: conditional execution descriptor (k, cond_iter, K, II, JJ, NT_I, NT_J)
  //   %tensor_desc: tensor position descriptor (m_pos/n_pos are tile indices, converted to elements internally)
  //   %load_memref: output memref[K, NT_I * NT_J] for returning variadic loaded
  //                 values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @simple_maybe_global_load_multi_tile
  func.func private @simple_maybe_global_load_multi_tile(
    %cond_desc: !conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2d,
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
    %ii = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, NT_I] -> (ii mod NT_I)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, NT_J] -> (jj mod NT_J)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_load {
      // Extract remaining fields from descriptor
      %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2d -> !sx2
      %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2d -> index

      // Load NT_I x NT_J tiles using simple 16x16 loads
      scf.for %i = %c0 to %NT_I step %c1 {
        scf.for %j = %c0 to %NT_J step %c1 {
          // Convert tile indices to element positions: (ii + i) * 16, (jj + j) * 16
          %m_pos = affine.apply affine_map<()[ii, i] -> ((ii + i) * 16)>()[%ii, %i]
          %n_pos = affine.apply affine_map<()[jj, j] -> ((jj + j) * 16)>()[%jj, %j]

          %pos_desc = aster_utils.struct_create(%ptr, %m_pos, %n_pos, %global_stride_in_bytes, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
          %value = func.call @simple_global_load_wave_16x16xf16_wait(%pos_desc)
            : (!tensor_position_descriptor_2d) -> !vx2

          %tile_idx = affine.apply affine_map<()[i, j, NT_J] -> (i * NT_J + j)>()[%i, %j, %NT_J]
          memref.store %value, %load_memref[%k, %tile_idx] : memref<?x?x!vx2>
        } {aster.constexpr}
      } {aster.constexpr}
    }

    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile global loads (coalesced)
  //   Executes bulk loads when alignment conditions are met
  //===--------------------------------------------------------------------===//

  // Multi-tile global load using global_load_wave_multi_tile_256xf16_via_dwordx2_wait.
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %cond_desc: conditional execution descriptor (k, cond_iter, K, II, JJ, NT_I, NT_J)
  //   %tensor_desc: 2-level tensor position descriptor where:
  //     - m_pos/n_pos are base positions in elements (major tile position)
  //     - mm_pos/nn_pos are tile indices (converted to elements internally)
  //   %load_memref: output memref[K, NT_I * NT_J] for returning variadic loaded
  //                 values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @maybe_global_load_multi_tile_coalesced
  func.func private @maybe_global_load_multi_tile_coalesced(
    %cond_desc: !conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from conditional execution descriptor
    %k = aster_utils.struct_extract %cond_desc["k"] : !conditional_execution_descriptor_2d -> index
    %cond_iter = aster_utils.struct_extract %cond_desc["cond_iter"] : !conditional_execution_descriptor_2d -> index
    %NT_I = aster_utils.struct_extract %cond_desc["NT_I"] : !conditional_execution_descriptor_2d -> index
    %NT_J = aster_utils.struct_extract %cond_desc["NT_J"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (mm_pos/nn_pos are tile indices here)
    %ii = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %jj = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index

    // Execute when cond_iter == 0 AND ii/jj are at NT boundaries
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, mult] -> (ii mod mult)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, mult] -> (jj mod mult)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_load {
      // Extract remaining fields from descriptor
      %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
      %m_pos_base = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
      %n_pos_base = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
      %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
      %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

      // Allocate 1D temp memref for multi-tile results (linearized)
      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] -> (NT_I * NT_J)>()[%NT_I, %NT_J]
      %temp_memref = memref.alloca(%NT_IJ) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Create 2-level descriptor for the bulk load primitive
      %load_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %ii_pos, %jj_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d

      // Load NT_I x NT_J tiles at once using bulk primitive
      func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
          %load_desc, %NT_I, %NT_J, %temp_memref)
        : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()

      // Copy results from temp memref to main memref (both use linearized indexing)
      scf.for %idx = %c0 to %NT_IJ step %c1 {
        %loaded = memref.load %temp_memref[%idx] : memref<?x!vx2>
        memref.store %loaded, %load_memref[%k, %idx] : memref<?x?x!vx2>
      } {aster.constexpr}
    }

    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile LDS writes (coalesced)
  //   Executes bulk writes when alignment conditions are met
  //===--------------------------------------------------------------------===//
  // Multi-tile LDS write using lds_write_wave_multi_tile_256xf16_via_dwordx2_wait.
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %cond_desc: conditional execution descriptor (k, cond_iter, K, II, JJ, NT_I, NT_J)
  //   %lds_desc: LDS position descriptor (m_pos/n_pos are tile indices, converted to elements internally)
  //   %load_memref: input memref[K, NT_I * NT_J] for reading variadic values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @maybe_lds_write_multi_tile_coalesced
  func.func private @maybe_lds_write_multi_tile_coalesced(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_desc: !lds_position_descriptor_2d,
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
    %ii = aster_utils.struct_extract %lds_desc["m_pos"] : !lds_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %lds_desc["n_pos"] : !lds_position_descriptor_2d -> index

    // Execute when cond_iter == 0 AND ii/jj are at NT boundaries
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, NT_I] -> (ii mod NT_I)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, NT_J] -> (jj mod NT_J)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_write {
      // Extract remaining fields from descriptor
      %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2d -> index
      %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2d -> index

      // Allocate 1D temp memref (linearized)
      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] -> (NT_I * NT_J)>()[%NT_I, %NT_J]
      %temp_memref = memref.alloca(%NT_IJ) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Copy results from main memref to temp memref using linearized indices
      scf.for %idx = %c0 to %NT_IJ step %c1 {
        %loaded = memref.load %load_memref[%k, %idx] : memref<?x?x!vx2>
        memref.store %loaded, %temp_memref[%idx] : memref<?x!vx2>
      } {aster.constexpr}

      // Create 2-level LDS descriptor for the bulk write primitive
      %lds_write_desc = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d

      // Write NT_I x NT_J tiles using bulk primitive
      func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
          %lds_write_desc, %NT_I, %NT_J, %temp_memref)
        : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
    }
    return
  }
}
