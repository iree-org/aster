// Conditional copy functions for AMDGCN kernels.
//
// Provides conditional primitives for GEMM kernels. All functions use the
// `maybe_` prefix indicating conditional execution:
// - C fragment initialization at first K iteration (k==0, kk==0)
// - C fragment store-to-global at last K iteration (k==K-1, kk==KK-1)
// - LDS read with tile reuse (execute when cond_zero == 0)
//
// Naming convention: @maybe_<operation>_wave_<tile_size>_<data_type>_<fragment_type>

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

// Future types from copies.mlir
!future_lds_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_write = !amdgcn.write_token<flat>

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - cond_zero: condition index (execute only when cond_zero == 0)
//   - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors (execute when ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0)
!conditional_execution_descriptor_2d = !aster_utils.struct<cond_zero: index, cond_mod_zero_i: index, cond_mod_zero_j: index>

// A 2D conditional execution descriptor for C fragment init/store operations containing:
//   - k, kk: current K tile indices (outer and inner)
//   - K, KK: total K tile counts (for first/last-iteration detection)
!store_conditional_execution_descriptor_2d = !aster_utils.struct<k: index, kk: index, K: index, KK: index>

// A 2D C fragment position descriptor containing:
//   - mm, nn: tile indices for C fragment indexing
!c_fragment_position_descriptor_2d = !aster_utils.struct<mm: index, nn: index>

//===-----------------------------------------------------------------------===//
// Wave-level, conditional C fragment init/store instructions, parameterizable by
// !store_conditional_execution_descriptor_2d and position descriptors.
//
// Conditionally initializes C fragments to zero (first K iteration) or stores
// C fragments to global memory (last K iteration) for GEMM K-loop patterns.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_c_fragment_init_store_single_wave isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // External function declarations
  //===--------------------------------------------------------------------===//

  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4

  // From copies.mlir
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

  //===--------------------------------------------------------------------===//
  // Conditional C fragment initialization
  //   Init C fragment to zero at first K iteration (k==0 AND kk==0)
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally initializes a C fragment (!vx4) to zero.
  // Executes only at first K iteration (k==0 AND kk==0) for GEMM reduction.
  //
  // Parameters:
  //   %cond_desc: !store_conditional_execution_descriptor_2d
  //     - k, kk: current K tile indices
  //     - K, KK: total K tile counts
  //   %pos_desc: !c_fragment_position_descriptor_2d
  //     - mm, nn: tile indices for C fragment indexing
  //   %c_fragments: memref<?x?x!vx4> - output memref to store initialized fragment
  func.func private @maybe_init_wave_16x16xf32_C_fragment(
    %cond_desc: !store_conditional_execution_descriptor_2d,
    %pos_desc: !c_fragment_position_descriptor_2d,
    %c_fragments: memref<?x?x!vx4>
  ) {
    %c0 = arith.constant 0 : index

    // Extract from conditional execution descriptor
    %k = aster_utils.struct_extract %cond_desc["k"] : !store_conditional_execution_descriptor_2d -> index
    %kk = aster_utils.struct_extract %cond_desc["kk"] : !store_conditional_execution_descriptor_2d -> index
    %K = aster_utils.struct_extract %cond_desc["K"] : !store_conditional_execution_descriptor_2d -> index
    %KK = aster_utils.struct_extract %cond_desc["KK"] : !store_conditional_execution_descriptor_2d -> index

    // Extract tile indices from position descriptor
    %mm = aster_utils.struct_extract %pos_desc["mm"] : !c_fragment_position_descriptor_2d -> index
    %nn = aster_utils.struct_extract %pos_desc["nn"] : !c_fragment_position_descriptor_2d -> index

    // Execute when k == 0 AND kk == 0 (first iteration of K reduction)
    %k_kk = affine.linearize_index [%k, %kk] by (%K, %KK) : index
    %is_first_k = arith.cmpi eq, %k_kk, %c0 : index

    scf.if %is_first_k {
      %c0_i32 = arith.constant 0 : i32
      %c_fragment = func.call @init_vgprx4(%c0_i32) : (i32) -> !vx4
      memref.store %c_fragment, %c_fragments[%mm, %nn] : memref<?x?x!vx4>
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional C fragment global store
  //   16x16xf32 C fragment stored at last K iteration (k==K-1 AND kk==KK-1)
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally stores a C fragment (16x16xf32) to global memory.
  // Executes only at last K iteration (k==K-1 AND kk==KK-1) for GEMM reduction.
  //
  // Parameters:
  //   %cond_desc: !store_conditional_execution_descriptor_2d
  //     - k, kk: current K tile indices
  //     - K, KK: total K tile counts
  //   %tensor_desc: !tensor_position_descriptor_2level_2d
  //     - ptr: base pointer
  //     - m_pos, n_pos: major tile position (element coordinates)
  //     - mm_pos, nn_pos: tile indices (converted to mm*16, nn*16 internally)
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (4 for f32)
  //   %c_fragments: memref<?x?x!vx4> - input memref with C fragments to store
  func.func private @maybe_global_store_wave_16x16xf32_C_fragment(
    %cond_desc: !store_conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %c_fragments: memref<?x?x!vx4>
  ) {
    %c1 = arith.constant 1 : index

    // Extract from conditional execution descriptor
    %k = aster_utils.struct_extract %cond_desc["k"] : !store_conditional_execution_descriptor_2d -> index
    %kk = aster_utils.struct_extract %cond_desc["kk"] : !store_conditional_execution_descriptor_2d -> index
    %K = aster_utils.struct_extract %cond_desc["K"] : !store_conditional_execution_descriptor_2d -> index
    %KK = aster_utils.struct_extract %cond_desc["KK"] : !store_conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (mm_pos/nn_pos are tile indices here)
    %mm = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %nn = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index

    // Execute when k == K-1 AND kk == KK-1 (last iteration of K reduction)
    %k_minus_1 = arith.subi %K, %c1 : index
    %kk_minus_1 = arith.subi %KK, %c1 : index
    %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
    %is_last_kk = arith.cmpi eq, %kk, %kk_minus_1 : index
    %should_store = arith.andi %is_last_k, %is_last_kk : i1

    scf.if %should_store {
      // Extract remaining fields from descriptor
      %c_global = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
      %m_pos = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
      %n_pos = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
      %GLOBAL_STRIDE_IN_BYTES = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
      %elt_size_c = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

      %fragment = memref.load %c_fragments[%mm, %nn] : memref<?x?x!vx4>

      // mm/nn are tile indices, so position = tile_index * 16
      %mm_pos = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
      %nn_pos = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]

      %pos_desc_c = aster_utils.struct_create(%c_global, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size_c) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
      %true_store = arith.constant true
      func.call @global_store_wave_16x16xf32_C_fragment_wait(%fragment, %pos_desc_c, %true_store) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional C fragment global store - FUTURE variant
  //   16x16xf32 C fragment stored at last K iteration (k==K-1 AND kk==KK-1)
  // (future variant - tracks execution state for consistency)
  //===--------------------------------------------------------------------===//
  // Conditionally stores a C fragment (16x16xf32) to global memory.
  // Executes only at last K iteration (k==K-1 AND kk==KK-1) for GEMM reduction.
  //
  // Note: Since C fragment store happens at the end of computation with no
  // subsequent work to overlap, this uses the _wait variant internally but
  // provides a consistent API with execution tracking via the return value.
  //
  // Parameters:
  //   %cond_desc: !store_conditional_execution_descriptor_2d - execution condition
  //   %tensor_desc: !tensor_position_descriptor_2level_2d - memory position
  //   %c_fragments: memref<?x?x!vx4> - input memref with C fragments to store
  // Returns:
  //   i1 - true if operation executed, false otherwise
  func.func private @maybe_global_store_wave_16x16xf32_C_fragment_future(
    %cond_desc: !store_conditional_execution_descriptor_2d,
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %c_fragments: memref<?x?x!vx4>
  ) -> i1 {
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true

    // Extract from conditional execution descriptor
    %k = aster_utils.struct_extract %cond_desc["k"] : !store_conditional_execution_descriptor_2d -> index
    %kk = aster_utils.struct_extract %cond_desc["kk"] : !store_conditional_execution_descriptor_2d -> index
    %K = aster_utils.struct_extract %cond_desc["K"] : !store_conditional_execution_descriptor_2d -> index
    %KK = aster_utils.struct_extract %cond_desc["KK"] : !store_conditional_execution_descriptor_2d -> index

    // Extract tile indices from descriptor (mm_pos/nn_pos are tile indices here)
    %mm = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %nn = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index

    // Execute when k == K-1 AND kk == KK-1 (last iteration of K reduction)
    %k_minus_1 = arith.subi %K, %c1 : index
    %kk_minus_1 = arith.subi %KK, %c1 : index
    %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
    %is_last_kk = arith.cmpi eq, %kk, %kk_minus_1 : index
    %should_store = arith.andi %is_last_k, %is_last_kk : i1

    %valid = scf.if %should_store -> i1 {
      // Extract remaining fields from descriptor
      %c_global = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
      %m_pos = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
      %n_pos = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
      %GLOBAL_STRIDE_IN_BYTES = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
      %elt_size_c = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

      %fragment = memref.load %c_fragments[%mm, %nn] : memref<?x?x!vx4>

      // mm/nn are tile indices, so position = tile_index * 16
      %mm_pos = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
      %nn_pos = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]

      %pos_desc_c = aster_utils.struct_create(%c_global, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size_c) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
      %true_store = arith.constant true
      // Use _wait variant since there's no subsequent computation to overlap with
      func.call @global_store_wave_16x16xf32_C_fragment_wait(%fragment, %pos_desc_c, %true_store) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }

    return %valid : i1
  }
}

//===-----------------------------------------------------------------------===//
// Wave-level, conditional LDS read instructions, parameterizable by
// !conditional_execution_descriptor_2d and !lds_position_descriptor_2d.
//
// Conditionally reads a 16x16xf16 fragment from LDS when cond_zero == 0.
// Used for GEMM A/B fragment loading with tile reuse across iterations.
//===-----------------------------------------------------------------------===//
amdgcn.library @conditional_lds_read_single_wave isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // External function declarations
  //===--------------------------------------------------------------------===//

  // From copies.mlir
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    !lds_position_descriptor_2d, i1) -> !vx2
  func.func private @lds_read_A_wave_16x16xf16_fragment_future(
    !lds_position_descriptor_2d, i1) -> !future_lds_read_any

  //===--------------------------------------------------------------------===//
  // Conditional LDS read
  //   16x16xf16 fragment read when cond_zero == 0
  // (conditional variant only)
  //===--------------------------------------------------------------------===//
  // Conditionally reads a 16x16xf16 fragment from LDS to VGPRs.
  // Executes only when cond_zero == 0 (for tile reuse across iterations).
  //
  // Usage pattern (GEMM):
  //   For A: ii=mm, jj=kk, cond_zero=nn -> memref[k, ii, jj]
  //   For B: ii=nn, jj=kk, cond_zero=mm -> memref[k, ii, jj]
  //   When cond_zero == 0: reads from LDS at (ii*16, jj*16), stores to memref
  //   When cond_zero != 0: no-op (value already in memref from cond_zero==0)
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d
  //     - cond_zero: execute only when == 0
  //     - cond_mod_zero_i, cond_mod_zero_j: unused (present for descriptor compatibility)
  //   %lds_pos_desc_base: !lds_position_descriptor_2d
  //     - lds_base: base offset in LDS (bytes)
  //     - m_pos, n_pos: unused (ii/jj used for position computation)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %k: outer loop index for memref storage
  //   %ii, %jj: inner tile indices for position computation and memref indexing
  //   %frag_memref: memref<?x?x?x!vx2> - 3D memref[K, II, JJ] for fragments
  func.func private @maybe_lds_read_wave_16x16xf16_fragment(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_pos_desc_base: !lds_position_descriptor_2d,
    %k: index, %ii: index, %jj: index,
    %frag_memref: memref<?x?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index

    // Extract from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index

    // Extract from LDS position descriptor base
    %lds_base_off = aster_utils.struct_extract %lds_pos_desc_base["lds_base"] : !lds_position_descriptor_2d -> index
    %LDS_STRIDE_IN_BYTES = aster_utils.struct_extract %lds_pos_desc_base["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
    %elt_size = aster_utils.struct_extract %lds_pos_desc_base["elt_size"] : !lds_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index

    scf.if %is_cond_zero {
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]

      %lds_pos_desc_read = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
      %false = arith.constant false
      %frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc_read, %false)
        : (!lds_position_descriptor_2d, i1) -> !vx2

      memref.store %frag, %frag_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Conditional LDS read - FUTURE variant
  //   16x16xf16 fragment read when cond_zero == 0
  // (future variant - returns without waiting)
  //===--------------------------------------------------------------------===//
  // Conditionally reads a 16x16xf16 fragment from LDS to VGPRs.
  // Uses @lds_read_A_wave_16x16xf16_fragment_future (non-blocking).
  // Executes only when cond_zero == 0 (for tile reuse across iterations).
  //
  // Parameters:
  //   %cond_desc: !conditional_execution_descriptor_2d - execution condition
  //   %lds_pos_desc_base: !lds_position_descriptor_2d - LDS position
  //   %k: index - outer loop index for memref storage
  //   %ii, %jj: index - inner tile indices for position computation
  //   %future_memref: memref<?x?x?x!future_lds_read_any> - output for futures
  // Returns:
  //   i1 - true if operation executed, false otherwise
  func.func private @maybe_lds_read_wave_16x16xf16_fragment_future(
    %cond_desc: !conditional_execution_descriptor_2d,
    %lds_pos_desc_base: !lds_position_descriptor_2d,
    %k: index, %ii: index, %jj: index,
    %future_memref: memref<?x?x?x!future_lds_read_any>
  ) -> i1 {
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %true = arith.constant true

    // Extract from conditional execution descriptor
    %cond_zero = aster_utils.struct_extract %cond_desc["cond_zero"] : !conditional_execution_descriptor_2d -> index

    // Extract from LDS position descriptor base
    %lds_base_off = aster_utils.struct_extract %lds_pos_desc_base["lds_base"] : !lds_position_descriptor_2d -> index
    %LDS_STRIDE_IN_BYTES = aster_utils.struct_extract %lds_pos_desc_base["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
    %elt_size = aster_utils.struct_extract %lds_pos_desc_base["elt_size"] : !lds_position_descriptor_2d -> index

    %is_cond_zero = arith.cmpi eq, %cond_zero, %c0 : index

    %valid = scf.if %is_cond_zero -> i1 {
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]

      %lds_pos_desc_read = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
      %transposed = arith.constant false
      %future = func.call @lds_read_A_wave_16x16xf16_fragment_future(%lds_pos_desc_read, %transposed)
        : (!lds_position_descriptor_2d, i1) -> !future_lds_read_any

      memref.store %future, %future_memref[%k, %ii, %jj] : memref<?x?x?x!future_lds_read_any>

      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }
    return %valid : i1
  }
}
