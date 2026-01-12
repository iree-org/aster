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

amdgcn.library @multi_tile_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // copies.mlir
  func.func private @simple_global_load_wave_16x16xf16_wait(!sx2, index, index, index) -> !vx2
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, index, index, index, index)
  func.func private @simple_lds_read_wave_16x16xf16_wait(index, index, index, index) -> !vx2
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index) -> !vx2
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(index, index, index, index, index, !vx2)

  //===--------------------------------------------------------------------===//
  // Simple conditional multi-tile global load
  //===--------------------------------------------------------------------===//
  // Simplified multi-tile global load using simple_global_load_wave_16x16xf16_wait.
  // Executes when cond_iter == 0 AND ii % NT_I == 0 AND jj % NT_J == 0.
  //
  // Parameters:
  //   %k: outer loop index (for storing variadic results in load_memref -> mem2reg)
  //   %ii, %jj: tile indices being iterated
  //   %cond_iter: condition index (execute only when cond_iter == 0)
  //   %K, %II, %JJ: loop bounds (unused but kept for API compatibility)
  //   %NT_I, %NT_J: multi-tile factors (load NT_I x NT_J tiles at once)
  //   %ptr: global memory pointer
  //   %i_pos_base, %j_pos_base: base positions in global memory
  //   %SIZE_J: stride in elements (converted to bytes internally)
  //   %load_memref: output memref[K, NT_I * NT_J] for returning variadic loaded
  //                 values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @maybe_global_load_multi_tile_simple
  func.func private @maybe_global_load_multi_tile_simple(
    %k: index, %ii: index, %jj: index, %cond_iter: index,
    %K: index, %II: index, %JJ: index,
    %NT_I: index, %NT_J: index,
    %ptr: !sx2,
    %i_pos_base: index, %j_pos_base: index, %SIZE_J: index,
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
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_load {
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      %elt_size = arith.constant 2 : index
      %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_J, elt_size] ->
        (SIZE_J * elt_size)>()[%SIZE_J, %elt_size]

      // Load NT_I x NT_J tiles using simple 16x16 loads
      scf.for %i = %c0 to %NT_I step %c1 {
        scf.for %j = %c0 to %NT_J step %c1 {
          %m_pos = affine.apply affine_map<()[i_pos_base, ii_pos, i] -> (i_pos_base + ii_pos + i * 16)>()[%i_pos_base, %ii_pos, %i]
          %n_pos = affine.apply affine_map<()[j_pos_base, jj_pos, j] -> (j_pos_base + jj_pos + j * 16)>()[%j_pos_base, %jj_pos, %j]

          %value = func.call @simple_global_load_wave_16x16xf16_wait(
            %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES)
            : (!sx2, index, index, index) -> !vx2

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
  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
    %ptr: !sx2,                      // Global base pointer
    %m_pos_base: index,              // Major-tile M position
    %n_pos_base: index,              // Major-tile N position
    %GLOBAL_STRIDE_IN_BYTES: index,  // Stride in bytes
    %mm_pos_base: index,             // Base minor-tile M position
    %nn_pos_base: index,             // Base minor-tile N position
    %m_tiles: index,                 // Number of tiles in M direction
    %n_tiles: index,                 // Number of tiles in N direction
    %result_memref: memref<?x!vx2>   // Output: m_tiles x n_tiles results
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

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
        %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(
          %ptr, %m_pos_base, %n_pos_base, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %row_size)
          : (!sx2, index, index, index, index, index, index) -> !vx2

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
  //   %k: outer loop index (for storing variadic results in load_memref -> mem2reg)
  //   %ii, %jj: tile indices being iterated
  //   %cond_iter: condition index (execute only when cond_iter == 0)
  //   %K, %II, %JJ: loop bounds (unused but kept for API compatibility)
  //   %NT_I, %NT_J: multi-tile factors (load NT_I x NT_J tiles at once)
  //   %ptr: global memory pointer
  //   %i_pos_base, %j_pos_base: base positions in global memory
  //   %SIZE_J: stride in elements (converted to bytes internally)
  //   %load_memref: output memref[K, NT_I * NT_J] for returning variadic loaded
  //                 values -> mem2reg.
  //
  // CHECK-LABEL: func.func private @maybe_global_load_multi_tile_coalesced
  func.func private @maybe_global_load_multi_tile_coalesced(
    %k: index, %ii: index, %jj: index, %cond_iter: index,
    %K: index, %II: index, %JJ: index,
    %NT_I: index, %NT_J: index,
    %ptr: !sx2,
    %i_pos_base: index, %j_pos_base: index, %SIZE_J: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Execute when cond_iter == 0 AND ii/jj are at NT boundaries
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, mult] -> (ii mod mult)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, mult] -> (jj mod mult)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_load {
      // Allocate temp memref for multi-tile results: [NT_I, NT_J]
      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] ->
        (NT_I * NT_J)>()[%NT_I, %NT_J]
      %temp_memref = memref.alloca(%NT_IJ) : memref<?x!vx2>

      %i_pos = affine.apply affine_map<()[i_pos_base, ii, NT_I] ->
        (i_pos_base)>()[%i_pos_base, %ii, %NT_I]
      %j_pos = affine.apply affine_map<()[j_pos_base, jj, NT_J] ->
        (j_pos_base)>()[%j_pos_base, %jj, %NT_J]

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      %elt_size = arith.constant 2 : index
      %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_J, elt_size] ->
        (SIZE_J * elt_size)>()[%SIZE_J, %elt_size]

      // Load NT_I x NT_J tiles at once using bulk primitive
      func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
          %ptr, %i_pos, %j_pos, %GLOBAL_STRIDE_IN_BYTES,
          %ii_pos, %jj_pos,
          %NT_I, %NT_J,
          %temp_memref)
        : (!sx2, index, index, index,
           index, index,
           index, index,
           memref<?x!vx2>) -> ()

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
  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    %lds_base_off: index,              // Base LDS offset
    %mm_pos_base: index,               // Base minor-tile M position
    %nn_pos_base: index,               // Base minor-tile N position
    %LDS_STRIDE_IN_BYTES: index,       // Stride in bytes
    %m_tiles: index,                   // Number of tiles in M direction
    %n_tiles: index,                   // Number of tiles in N direction
    %values_memref: memref<?x!vx2>   // Input: m_tiles x n_tiles values
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

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
        func.call @lds_write_wave_256xf16_via_dwordx2_wait(
          %lds_base_off, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %row_size, %value)
          : (index, index, index, index, index, !vx2) -> ()
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Multi-tile LDS write using lds_write_wave_multi_tile_256xf16_via_dwordx2_wait.
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
  // CHECK-LABEL: func.func private @maybe_lds_write_multi_tile_coalesced
  func.func private @maybe_lds_write_multi_tile_coalesced(
    %k: index, %ii: index, %jj: index, %cond_iter: index,
    %K: index, %II: index, %JJ: index,
    %NT_I: index, %NT_J: index,
    %lds_base_off: index,
    %SIZE_J: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Execute when cond_iter == 0 AND ii/jj are at NT boundaries
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, NT_I] -> (ii mod NT_I)>()[%ii, %NT_I]
    %jj_mod = affine.apply affine_map<()[jj, NT_J] -> (jj mod NT_J)>()[%jj, %NT_J]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    scf.if %should_write {
      %NT_IJ = affine.apply affine_map<()[NT_I, NT_J] ->
        (NT_I * NT_J)>()[%NT_I, %NT_J]
      %temp_memref = memref.alloca(%NT_IJ) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      %elt_size = arith.constant 2 : index
      %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_J, elt_size] ->
        (SIZE_J * elt_size)>()[%SIZE_J, %elt_size]

      // Copy results from main memref to temp memref
      scf.for %idx = %c0 to %NT_IJ step %c1 {
        %loaded = memref.load %load_memref[%k, %idx] : memref<?x?x!vx2>
        memref.store %loaded, %temp_memref[%idx] : memref<?x!vx2>
      } {aster.constexpr}

      // Write NT_I x NT_J tiles using bulk primitive
      func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
          %lds_base_off, %ii_pos, %jj_pos, %LDS_STRIDE_IN_BYTES,
          %NT_I, %NT_J,
          %temp_memref)
        : (index, index, index, index, index, index, memref<?x!vx2>) -> ()
    }
    return
  }
}
