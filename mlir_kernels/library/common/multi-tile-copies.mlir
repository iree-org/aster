// Multi-tile copy functions for AMDGCN kernels.
// These functions handle conditional multi-tile global loads and LDS writes
// using the simpler 16x16 wave-level primitives.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/simple-copies.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr_range<[? + 1]>
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!a   = !amdgcn.agpr
!ax1 = !amdgcn.agpr_range<[? + 1]>
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

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

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

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
  // Simple conditional multi-tile global load
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
  // Conditional coalesced multi-tile global read
  //===--------------------------------------------------------------------===//
  // Helper function: Multi-tile version of global_load_wave_256xf16_via_dwordx2_wait.
  // Loads m_tiles x n_tiles 16x16 tiles from global memory.
  // Results are stored in a provided memref of shape [m_tiles, n_tiles].
  //
  // This function enables loading larger regions (e.g., 16x64 = 4 tiles)
  // for better memory coalescing while preserving the scheduling attributes.
  // Each tile load includes its own waitcnt 0 (simpler wait strategy).
  //
  // Parameters:
  //   %tensor_desc: 2-level tensor position descriptor where:
  //     - m_pos/n_pos are major-tile positions (in elements)
  //     - mm_pos/nn_pos are base minor-tile positions (in elements)
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %result_memref: output memref[m_tiles * n_tiles] for results
  //
  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %m_tiles: index,                 // Number of tiles in M direction
    %n_tiles: index,                 // Number of tiles in N direction
    %result_memref: memref<?x!vx2>   // Output: m_tiles x n_tiles results
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from descriptor
    %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
    %m_pos_base = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
    %n_pos_base = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
    %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

    // Each transfer does row_size * col_size elements, this is a reshape via a
    // 256-size tile with a number of rows that is determined internally by
    // global_load_wave_256xf16_via_dwordx2_wait.
    %row_size = affine.apply affine_map<()[n_tiles] -> (16 ceildiv n_tiles)>()[%n_tiles]
    %col_size = affine.apply affine_map<()[n_tiles] -> (16 * n_tiles)>()[%n_tiles]
    %total_rows = affine.apply affine_map<()[m_tiles] -> (16 * m_tiles)>()[%m_tiles]
    %total_cols = affine.apply affine_map<()[n_tiles] -> (16 * n_tiles)>()[%n_tiles]

    scf.for %mt = %c0 to %total_rows step %row_size {
      scf.for %nt = %c0 to %total_cols step %col_size {
        // Compute minor-tile positions for this tile
        // NOTE: Only add mt/nt to the minor positions (mm_pos/nn_pos), NOT to major
        // positions (m_pos/n_pos). tiledx2_matrix_offset adds them together, so
        // adding to both would double-count the offset.
        %mm_pos = affine.apply affine_map<()[base, mt] -> (base + mt)>()[%mm_pos_base, %mt]
        %nn_pos = affine.apply affine_map<()[base, nt] -> (base + nt)>()[%nn_pos_base, %nt]

        // Load the tile
        %pos_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %mm_pos, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %transfer_size = arith.constant 8 : index // dwordx2
        %wave_size = arith.constant 64 : index
        %transfer_desc = aster_utils.struct_create(%row_size, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
        %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2

        // Store result in memref
        %i = affine.apply affine_map<()[mt, row_size] -> (mt ceildiv row_size)>()[%mt, %row_size]
        %j = affine.apply affine_map<()[nt, col_size] -> (nt ceildiv col_size)>()[%nt, %col_size]
        %J = affine.apply affine_map<()[total_cols, col_size] -> (total_cols ceildiv col_size)>()[%total_cols, %col_size]
        %idx = affine.apply affine_map<()[i, j, J] -> (i * J + j)>()[%i, %j, %J]
        memref.store %loaded, %result_memref[%idx] : memref<?x!vx2>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

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

      // Allocate temp memref for multi-tile results: [NT_I, NT_J]
      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] ->
        (NT_I * NT_J)>()[%NT_I, %NT_J]
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

      // Copy results from temp memref to main memref
      scf.for %idx = %c0 to %NT_IJ step %c1 {
        %loaded = memref.load %temp_memref[%idx] : memref<?x!vx2>
        memref.store %loaded, %load_memref[%k, %idx] : memref<?x?x!vx2>
      } {aster.constexpr}
    }

    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional coalesced multi-tile LDS write
  //===--------------------------------------------------------------------===//
  // Multi-tile version of lds_write_wave_256xf16_via_dwordx2_wait.
  // Writes m_tiles x n_tiles 16x16 tiles to LDS.
  // Values are read from a provided memref of shape [m_tiles, n_tiles].
  //
  // This function enables writing larger regions (e.g., 16x64 = 4 tiles)
  // for connecting to better memory coalescing global loads multi-tile version.
  // Each tile write includes its own waitcnt 0 (simpler wait strategy).
  //
  // Parameters:
  //   %lds_desc: 2-level LDS position descriptor where:
  //     - lds_base is the base LDS offset
  //     - mm_pos/nn_pos are base minor-tile positions (in elements)
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %values_memref: input memref[m_tiles * n_tiles] with values to write
  //
  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,                   // Number of tiles in M direction
    %n_tiles: index,                   // Number of tiles in N direction
    %values_memref: memref<?x!vx2>     // Input: m_tiles x n_tiles values
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from descriptor
    %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %lds_desc["mm_pos"] : !lds_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %lds_desc["nn_pos"] : !lds_position_descriptor_2level_2d -> index
    %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2level_2d -> index

    // Each transfer does row_size * col_size elements, this is a reshape via a
    // 256-size tile with a number of rows that is determined internally by
    // lds_write_wave_256xf16_via_dwordx2_wait.
    %row_size = affine.apply affine_map<()[n_tiles] -> (16 ceildiv n_tiles)>()[%n_tiles]
    %col_size = affine.apply affine_map<()[n_tiles] -> (16 * n_tiles)>()[%n_tiles]
    %total_rows = affine.apply affine_map<()[m_tiles] -> (16 * m_tiles)>()[%m_tiles]
    %total_cols = affine.apply affine_map<()[n_tiles] -> (16 * n_tiles)>()[%n_tiles]

    scf.for %mt = %c0 to %total_rows step %row_size {
      scf.for %nt = %c0 to %total_cols step %col_size {
        // Load value from memref
        %i = affine.apply affine_map<()[mt, row_size] -> (mt ceildiv row_size)>()[%mt, %row_size]
        %j = affine.apply affine_map<()[nt, col_size] -> (nt ceildiv col_size)>()[%nt, %col_size]
        %J = affine.apply affine_map<()[total_cols, col_size] -> (total_cols ceildiv col_size)>()[%total_cols, %col_size]
        %idx = affine.apply affine_map<()[i, j, J] -> (i * J + j)>()[%i, %j, %J]
        %value = memref.load %values_memref[%idx] : memref<?x!vx2>

        // Compute minor-tile positions for this tile
        %mm_pos = affine.apply affine_map<()[base, mt] -> (base + mt)>()[%mm_pos_base, %mt]
        %nn_pos = affine.apply affine_map<()[base, nt] -> (base + nt)>()[%nn_pos_base, %nt]

        // Write the tile to LDS
        %lds_pos_desc = aster_utils.struct_create(%lds_base_off, %mm_pos, %nn_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_size_lds = arith.constant 8 : index // dwordx2
        %wave_size_lds = arith.constant 64 : index
        %transfer_desc_lds = aster_utils.struct_create(%row_size, %transfer_size_lds, %wave_size_lds) : (index, index, index) -> !transfer_descriptor_2d
        func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc, %transfer_desc_lds, %value)
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

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

      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] ->
        (NT_I * NT_J)>()[%NT_I, %NT_J]
      %temp_memref = memref.alloca(%NT_IJ) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Copy results from main memref to temp memref
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
