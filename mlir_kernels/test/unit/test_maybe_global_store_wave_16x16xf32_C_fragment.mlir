// Unit test for maybe_global_store_wave_16x16xf32_C_fragment from conditional-copies.mlir
// Tests that C fragment is stored only at the last K iteration (k == K-1 AND kk == KK-1)

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!store_conditional_execution_descriptor_2d = !aster_utils.struct<k: index, kk: index, K: index, KK: index>

amdgcn.module @test_maybe_global_store_wave_16x16xf32_C_fragment target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From register-init.mlir
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // From indexing.mlir
  func.func private @lane_id() -> index
  // From conditional-copies.mlir
  func.func private @maybe_global_store_wave_16x16xf32_C_fragment(!store_conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx4>)

  //===--------------------------------------------------------------------===//
  // Test maybe_global_store_wave_16x16xf32_C_fragment: store only at last K iteration
  // Tests that the function correctly stores C fragment only when
  // k == K-1 AND kk == KK-1
  //===--------------------------------------------------------------------===//
  // Setup: MM=2, NN=2 (4 fragments), K=2, KK=2 (4 total K iterations)
  // Expected: fragments stored only at (k=1, kk=1)
  amdgcn.kernel @test_maybe_global_store_wave_16x16xf32_C_fragment arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Tile dimensions: MM x NN tiles of 16x16
    %MM = arith.constant {{MM}} : index
    %NN = arith.constant {{NN}} : index
    %K = arith.constant {{K}} : index
    %KK = arith.constant {{KK}} : index

    // Global stride in bytes: (NN * 16) elements * 4 bytes (f32)
    %global_stride_bytes = arith.constant {{GLOBAL_STRIDE_BYTES}} : index
    %elt_size = arith.constant 4 : index

    // Allocate C fragments memref[MM, NN]
    %c_fragments = memref.alloca(%MM, %NN) : memref<?x?x!vx4>

    // Initialize all C fragments with lane_id
    %lane = func.call @lane_id() : () -> index
    %lane_i32 = arith.index_cast %lane : index to i32
    %lane_reg = lsir.to_reg %lane_i32 : i32 -> !v
    %acc = func.call @init_vgprx4_reg(%lane_reg) : (!v) -> !vx4

    // Store initial values to C fragments
    scf.for %mm = %c0 to %MM step %c1 {
      scf.for %nn = %c0 to %NN step %c1 {
        memref.store %acc, %c_fragments[%mm, %nn] : memref<?x?x!vx4>
      } {aster.constexpr}
    } {aster.constexpr}

    // Loop over all (k, kk) iterations and call maybe_global_store_wave_16x16xf32_C_fragment for each fragment
    // The function should only store at the last iteration (k=K-1, kk=KK-1)
    scf.for %k = %c0 to %K step %c1 {
      scf.for %kk = %c0 to %KK step %c1 {
        scf.for %mm = %c0 to %MM step %c1 {
          scf.for %nn = %c0 to %NN step %c1 {
            // Create conditional execution descriptor
            %cond_desc = aster_utils.struct_create(%k, %kk, %K, %KK) : (index, index, index, index) -> !store_conditional_execution_descriptor_2d

            // Create tensor descriptor with mm/nn as tile indices
            %tensor_desc = aster_utils.struct_create(%out_ptr, %c0, %c0, %global_stride_bytes, %mm, %nn, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d

            // Call maybe_global_store_wave_16x16xf32_C_fragment (should only store at k=1, kk=1)
            func.call @maybe_global_store_wave_16x16xf32_C_fragment(%cond_desc, %tensor_desc, %c_fragments)
              : (!store_conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx4>) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
