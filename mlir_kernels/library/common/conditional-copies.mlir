// Conditional copy functions for AMDGCN kernels.
//
// Provides conditional C fragment initialization and store-to-global primitives.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D conditional execution descriptor for C fragment init/store operations containing:
//   - k, kk: current K tile indices (outer and inner)
//   - K, KK: total K tile counts (for first/last-iteration detection)
!store_conditional_execution_descriptor_2d = !aster_utils.struct<k: index, kk: index, K: index, KK: index>

// A 2D C fragment position descriptor containing:
//   - mm, nn: tile indices for C fragment indexing
!c_fragment_position_descriptor_2d = !aster_utils.struct<mm: index, nn: index>

amdgcn.library @conditional_copies isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // External function declarations
  //===--------------------------------------------------------------------===//

  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4

  // From simple-copies.mlir
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

  //===--------------------------------------------------------------------===//
  // Conditional C fragment initialization
  //   Initialize C fragments to zero at first K iteration
  //===--------------------------------------------------------------------===//

  // Conditionally initialize C fragment to zero.
  // Executes only when k == 0 AND kk == 0 (first iteration of K reduction).
  //
  // Parameters:
  //   %cond_desc: conditional execution descriptor (k, kk, K, KK)
  //   %pos_desc: C fragment position descriptor (mm, nn tile indices)
  //   %c_fragments: output memref[MM, NN] of C fragments to initialize
  //
  func.func private @maybe_init_C(
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
  // Conditional C fragment global stores
  //   16x16xf32 C fragments stored only at last K iteration
  //===--------------------------------------------------------------------===//

  // Conditionally store C fragment to global memory.
  // Executes only when k == K-1 AND kk == KK-1 (last iteration of K reduction).
  //
  // Parameters:
  //   %cond_desc: conditional execution descriptor (k, kk, K, KK)
  //   %tensor_desc: 2-level tensor position descriptor where:
  //     - m_pos/n_pos are base positions in elements (major tile position)
  //     - mm_pos/nn_pos are tile indices (converted to elements internally)
  //   %c_fragments: input memref[MM, NN] of C fragments to store
  //
  func.func private @maybe_store_c_fragment(
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
}
