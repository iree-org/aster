// Conditional multi-tile copy functions for AMDGCN kernels.
//
// Provides conditional variants of the multi-tile copy primitives in multi-tile-copies.mlir.
// All functions use the `maybe_` prefix indicating conditional execution.
// Operations execute based on alignment conditions:
// - cond_zero == 0 (execute at specified iteration)
// - ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0 (tile alignment for multi-tile batching)
//
// Naming convention: @maybe_<operation>_wave_multi_tile_<data_size>

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

// Future types from copies.mlir
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!future_lds_write = !amdgcn.write_token<shared>

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - cond_zero: condition index (execute only when cond_zero == 0)
//   - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors (execute when ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0)
!conditional_execution_descriptor_2d = !aster_utils.struct<cond_zero: index, cond_mod_zero_i: index, cond_mod_zero_j: index>

//===-----------------------------------------------------------------------===//
// Wave-level, conditional multi-tile global load instructions (coalesced),
// parameterizable by !conditional_execution_descriptor_2d and !tensor_position_descriptor_2level_2d.
//
// Conditionally loads cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles using bulk coalesced primitive.
// Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_multi_tile_global_load_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From multi-tile-copies.mlir
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future(!tensor_position_descriptor_2level_2d, index, index, memref<?x!future_global_read_any>) -> ()

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile global load (coalesced)
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via bulk dwordx2 load
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally loads cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from global memory.
  // Uses @global_load_wave_multi_tile_256xf16_via_dwordx2_wait (coalesced).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - cond_zero: execute only when == 0
  //     - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors
  //   %tensor_desc: !tensor_position_descriptor_2level_2d
  //     - ptr: base pointer
  //     - m_pos, n_pos: major tile position (element coordinates)
  //     - mm_pos, nn_pos: tile indices (converted to mm*16, nn*16 internally)
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %k: loop index for memref storage
  //   %load_memref: memref<?x?x!vx2> - output memref[K, cond_mod_zero_i * cond_mod_zero_j]

  // CHECK-LABEL: func.func private @maybe_global_load_wave_multi_tile_256xf16
  func.func private @maybe_global_load_wave_multi_tile_256xf16(
    %cond_desc: !conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %k: index,
    %load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract fields from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_i = aster_utils.struct_extract %cond_desc["cond_mod_zero_i"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_j = aster_utils.struct_extract %cond_desc["cond_mod_zero_j"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (mm_pos/nn_pos are tile indices here)
    %ii = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %jj = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index

    // Execute when cond_zero == 0 AND ii/jj are at boundaries
    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, mult] -> (ii mod mult)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, mult] -> (jj mod mult)>()[%jj, %cond_mod_zero_j]
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
      %num_tiles = affine.apply affine_map<()[i, j] -> (i * j)>()[%cond_mod_zero_i, %cond_mod_zero_j]
      %temp_memref = memref.alloca(%num_tiles) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Create 2-level descriptor for the bulk load primitive
      %load_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %ii_pos, %jj_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d

      // Load cond_mod_zero_i x cond_mod_zero_j tiles at once using bulk primitive
      func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
          %load_desc, %cond_mod_zero_i, %cond_mod_zero_j, %temp_memref)
        : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()

      // Copy results from temp memref to main memref (both use linearized indexing)
      scf.for %idx = %c0 to %num_tiles step %c1 {
        %loaded = memref.load %temp_memref[%idx] : memref<?x!vx2>
        memref.store %loaded, %load_memref[%k, %idx] : memref<?x?x!vx2>
      } {aster.constexpr}
    }

    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile global load (coalesced) - FUTURE variant
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via bulk dwordx2 load
  // (future variant - returns without waiting)
  //===--------------------------------------------------------------------===//
  // Conditionally loads cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from global memory.
  // Uses @global_load_wave_multi_tile_256xf16_via_dwordx2_future (non-blocking).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d - execution condition
  //   %tensor_desc: !tensor_position_descriptor_2level_2d - memory position
  //   %k: loop index for memref storage
  //   %load_memref: memref<?x?x!vx2> - output memref[K, cond_mod_zero_i * cond_mod_zero_j] for values
  //   %future_memref: memref<?x!future_global_read_any> - output for futures
  // Returns:
  //   i1 - true if operation executed, false otherwise

  // CHECK-LABEL: func.func private @maybe_global_load_wave_multi_tile_256xf16_future
  func.func private @maybe_global_load_wave_multi_tile_256xf16_future(
    %cond_desc: !conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %k: index,
    %load_memref: memref<?x?x!vx2>,
    %future_memref: memref<?x!future_global_read_any>
  ) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true

    // Extract fields from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_i = aster_utils.struct_extract %cond_desc["cond_mod_zero_i"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_j = aster_utils.struct_extract %cond_desc["cond_mod_zero_j"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (mm_pos/nn_pos are tile indices here)
    %ii = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %jj = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index

    // Execute when cond_zero == 0 AND ii/jj are at boundaries
    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, mult] -> (ii mod mult)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, mult] -> (jj mod mult)>()[%jj, %cond_mod_zero_j]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_load = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    %valid = scf.if %should_load -> i1 {
      // Extract remaining fields from descriptor
      %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
      %m_pos_base = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
      %n_pos_base = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
      %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
      %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Create 2-level descriptor for the bulk load primitive
      %load_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %ii_pos, %jj_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d

      // Load cond_mod_zero_i x cond_mod_zero_j tiles at once using bulk future primitive (non-blocking)
      func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
          %load_desc, %cond_mod_zero_i, %cond_mod_zero_j, %future_memref)
        : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!future_global_read_any>) -> ()

      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }

    return %valid : i1
  }
}

//===-----------------------------------------------------------------------===//
// Wave-level, conditional multi-tile LDS write instructions (coalesced),
// parameterizable by !conditional_execution_descriptor_2d and !lds_position_descriptor_2d.
//
// Conditionally writes cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles using bulk coalesced primitive.
// Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_multi_tile_lds_write_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From multi-tile-copies.mlir
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>, memref<?x!future_lds_write>) -> ()

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile LDS write (coalesced)
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via bulk ds_write_b64
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally writes cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from VGPRs to LDS.
  // Uses @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait (coalesced).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - cond_zero: execute only when == 0
  //     - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors
  //   %lds_desc: !lds_position_descriptor_2d
  //     - lds_base: base offset in LDS (bytes)
  //     - m_pos, n_pos: tile indices (converted to m*16, n*16 internally)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %k: loop index for memref access
  //   %load_memref: memref<?x?x!vx2> - input memref[K, cond_mod_zero_i * cond_mod_zero_j]

  // CHECK-LABEL: func.func private @maybe_lds_write_wave_multi_tile_256xf16
  func.func private @maybe_lds_write_wave_multi_tile_256xf16(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_desc: !lds_position_descriptor_2d,
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
    %ii = aster_utils.struct_extract %lds_desc["m_pos"] : !lds_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %lds_desc["n_pos"] : !lds_position_descriptor_2d -> index

    // Execute when cond_zero == 0 AND ii/jj are at boundaries
    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, cond_mod_zero_i] -> (ii mod cond_mod_zero_i)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, cond_mod_zero_j] -> (jj mod cond_mod_zero_j)>()[%jj, %cond_mod_zero_j]
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
      %num_tiles = affine.apply affine_map<()[i, j] -> (i * j)>()[%cond_mod_zero_i, %cond_mod_zero_j]
      %temp_memref = memref.alloca(%num_tiles) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Copy results from main memref to temp memref using linearized indices
      scf.for %idx = %c0 to %num_tiles step %c1 {
        %loaded = memref.load %load_memref[%k, %idx] : memref<?x?x!vx2>
        memref.store %loaded, %temp_memref[%idx] : memref<?x!vx2>
      } {aster.constexpr}

      // Create 2-level LDS descriptor for the bulk write primitive
      %lds_write_desc = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d

      // Write cond_mod_zero_i x cond_mod_zero_j tiles using bulk primitive
      func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
          %lds_write_desc, %cond_mod_zero_i, %cond_mod_zero_j, %temp_memref)
        : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional multi-tile LDS write (coalesced) - FUTURE variant
  //   cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles via bulk ds_write_b64
  // (future variant - returns without waiting)
  //===--------------------------------------------------------------------===//
  // Conditionally writes cond_mod_zero_i x cond_mod_zero_j 16x16xf16 tiles from VGPRs to LDS.
  // Uses @lds_write_wave_multi_tile_256xf16_via_dwordx2_future (non-blocking).
  // Executes when cond_zero == 0 AND ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0.
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d - execution condition
  //   %lds_desc: !lds_position_descriptor_2d - LDS position
  //   %k: loop index for memref access
  //   %load_memref: memref<?x?x!vx2> - input memref[K, cond_mod_zero_i * cond_mod_zero_j] with values
  //   %token_memref: memref<?x!future_lds_write> - output for write tokens
  // Returns:
  //   i1 - true if operation executed, false otherwise

  // CHECK-LABEL: func.func private @maybe_lds_write_wave_multi_tile_256xf16_future
  func.func private @maybe_lds_write_wave_multi_tile_256xf16_future(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_desc: !lds_position_descriptor_2d,
    %k: index,
    %load_memref: memref<?x?x!vx2>,
    %token_memref: memref<?x!future_lds_write>
  ) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true

    // Extract fields from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_i = aster_utils.struct_extract %cond_desc["cond_mod_zero_i"] : !conditional_execution_descriptor_2d -> index
    %cond_mod_zero_j = aster_utils.struct_extract %cond_desc["cond_mod_zero_j"] : !conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (m_pos/n_pos are tile indices here)
    %ii = aster_utils.struct_extract %lds_desc["m_pos"] : !lds_position_descriptor_2d -> index
    %jj = aster_utils.struct_extract %lds_desc["n_pos"] : !lds_position_descriptor_2d -> index

    // Execute when cond_zero == 0 AND ii/jj are at boundaries
    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index
    %ii_mod = affine.apply affine_map<()[ii, cond_mod_zero_i] -> (ii mod cond_mod_zero_i)>()[%ii, %cond_mod_zero_i]
    %jj_mod = affine.apply affine_map<()[jj, cond_mod_zero_j] -> (jj mod cond_mod_zero_j)>()[%jj, %cond_mod_zero_j]
    %is_ii_aligned = arith.cmpi eq, %ii_mod, %c0 : index
    %is_jj_aligned = arith.cmpi eq, %jj_mod, %c0 : index
    %ii_jj_aligned = arith.andi %is_ii_aligned, %is_jj_aligned : i1
    %should_write = arith.andi %is_cond_zero, %ii_jj_aligned : i1

    %valid = scf.if %should_write -> i1 {
      // Extract remaining fields from descriptor
      %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2d -> index
      %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
      %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2d -> index

      // Allocate 1D temp memref (linearized)
      %num_tiles = affine.apply affine_map<()[i, j] -> (i * j)>()[%cond_mod_zero_i, %cond_mod_zero_j]
      %temp_memref = memref.alloca(%num_tiles) : memref<?x!vx2>

      // ii/jj are tile indices, so position = tile_index * 16
      %ii_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

      // Copy results from main memref to temp memref using linearized indices
      scf.for %idx = %c0 to %num_tiles step %c1 {
        %loaded = memref.load %load_memref[%k, %idx] : memref<?x?x!vx2>
        memref.store %loaded, %temp_memref[%idx] : memref<?x!vx2>
      } {aster.constexpr}

      // Create 2-level LDS descriptor for the bulk write primitive
      %lds_write_desc = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d

      // Write cond_mod_zero_i x cond_mod_zero_j tiles using bulk future primitive (non-blocking)
      func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(
          %lds_write_desc, %cond_mod_zero_i, %cond_mod_zero_j, %temp_memref, %token_memref)
        : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>, memref<?x!future_lds_write>) -> ()

      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }

    return %valid : i1
  }
}
