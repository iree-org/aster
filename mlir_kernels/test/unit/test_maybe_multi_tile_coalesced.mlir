// Unit test for maybe_global_load_multi_tile_coalesced and maybe_lds_write_multi_tile_coalesced
// Tests the GEMM-style bulk multi-tile pattern where operations execute when
// ii % NT_I == 0 AND jj % NT_J == 0

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

amdgcn.module @test_maybe_multi_tile_coalesced target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From simple-copies.mlir
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(!lds_position_descriptor_2d, !tensor_position_descriptor_2d)
  // From multi-tile-copies.mlir
  func.func private @maybe_global_load_multi_tile_coalesced(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)

  //===--------------------------------------------------------------------===//
  // Test maybe_*_multi_tile_coalesced pattern from GEMM (bulk version)
  // This tests the bulk multi-tile library functions from multi_tile_copies.mlir
  //===--------------------------------------------------------------------===//
  // Pattern: Loop over (ii, jj) indices, execute multi-tile load/write when
  // ii % NT_I == 0 AND jj % NT_J == 0
  // Input: 32x64 array (2x4 tiles of 16x16)
  // Test with configurable NT_I, NT_J to exercise multiple batches
  amdgcn.kernel @test_maybe_multi_tile_coalesced arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 16384 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters: configurable via preprocessing (II x JJ tiles, load NT_I x NT_J tiles at a time)
    %SIZE_J = arith.constant {{SIZE_J}} : index
    %K = arith.constant 1 : index    // Outer loop size (single iteration for simplicity)
    %II = arith.constant {{II}} : index   // Total tiles in I dimension
    %JJ = arith.constant {{JJ}} : index   // Total tiles in J dimension
    %NT_I = arith.constant {{NT_I}} : index // Multi-tile factor I
    %NT_J = arith.constant {{NT_J}} : index // Multi-tile factor J
    %global_stride_bytes_coal = arith.constant {{GLOBAL_STRIDE_BYTES}} : index // SIZE_J * 2 bytes

    // Allocate 2D memref for library functions: [K, NT_I*NT_J]
    // This specific shape is required by `maybe_global_load_multi_tile_coalesced`
    // and by `maybe_lds_write_multi_tile_coalesced`.
    %load_memref_static = memref.alloca() : memref<1x{{NT_PRODUCT}}x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<1x{{NT_PRODUCT}}x!vx2> to memref<?x?x!vx2>

    // Loop over all tile indices like in GEMM (single k iteration)
    %elt_size_global_coal = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Create conditional execution descriptor (k=0, cond_iter=0 to always execute when aligned)
        %cond_desc_coal = aster_utils.struct_create(%c0, %c0, %NT_I, %NT_J) : (index, index, index, index) -> !conditional_execution_descriptor_2d

        // Call library function for global load
        // 2-level descriptor: m_pos/n_pos=0 (base positions), mm_pos/nn_pos=ii/jj (tile indices)
        %tensor_desc_coal = aster_utils.struct_create(%in_ptr, %c0, %c0, %global_stride_bytes_coal, %ii, %jj, %elt_size_global_coal) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        func.call @maybe_global_load_multi_tile_coalesced(
          %cond_desc_coal,              // conditional_execution_descriptor_2d
          %tensor_desc_coal,            // tensor_position_descriptor_2level_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write
        // LDS descriptor: lds_base=0, m_pos=ii, n_pos=jj (tile indices)
        %lds_stride_bytes_coal = arith.constant {{GLOBAL_STRIDE_BYTES}} : index // SIZE_J * 2 bytes
        %elt_size_lds_coal = arith.constant 2 : index
        %lds_desc_coal = aster_utils.struct_create(%c0, %ii, %jj, %lds_stride_bytes_coal, %elt_size_lds_coal) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        func.call @maybe_lds_write_multi_tile_coalesced(
          %cond_desc_coal,              // conditional_execution_descriptor_2d
          %lds_desc_coal,               // lds_position_descriptor_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant {{GLOBAL_STRIDE_BYTES}} : index // SIZE_J * 2 bytes
    %elt_size_coal = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        %lds_pos_coal = aster_utils.struct_create(%c0, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_coal) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %global_pos_coal = aster_utils.struct_create(%out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_coal) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_coal, %global_pos_coal)
          : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
