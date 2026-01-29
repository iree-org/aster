// Unit test for simple_maybe_global_load_multi_tile and simple_maybe_lds_write_multi_tile
// Tests the GEMM-style multi-tile pattern where operations execute when
// ii % NT_I == 0 AND jj % NT_J == 0

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!index_pair = !aster_utils.struct<i: index, j: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
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

amdgcn.module @test_maybe_multi_tile_simple target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  // From simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!tensor_position_descriptor_2d, !lds_position_descriptor_2d)
  func.func private @simple_global_store_wave_16x16xf16_wait(!vx2, !tensor_position_descriptor_2d)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(!lds_position_descriptor_2d, !tensor_position_descriptor_2d)
  // From copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(!tensor_position_descriptor_2level_2d, index, index)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d, i1) -> !vx2
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !tensor_position_descriptor_2level_2d, i1)
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, index, index, memref<?x?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, index, index, memref<?x?x!vx2>)
  // From simple-multi-tile-copies.mlir
  func.func private @simple_maybe_lds_write_multi_tile(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)
  // From multi-tile-copies.mlir
  func.func private @simple_maybe_global_load_multi_tile(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2d, memref<?x?x!vx2>)
  func.func private @maybe_global_load_multi_tile_coalesced(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)

  //===--------------------------------------------------------------------===//
  // Test maybe_*_multi_tile_simple pattern from GEMM
  // This tests the library functions from multi_tile_copies.mlir
  //===--------------------------------------------------------------------===//
  // Pattern: Loop over (ii, jj) indices, execute multi-tile load/write when
  // ii % NT_I == 0 AND jj % NT_J == 0
  // Input: 64x64 array (4x4 tiles of 16x16)
  // Test with NT_I=2, NT_J=2 to exercise multiple batches
  amdgcn.kernel @test_maybe_multi_tile_simple arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 16384 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %SIZE_J = arith.constant 128 : index

    // Parameters: 4x8 tiles, load 2x4 tiles at a time
    %K = arith.constant 1 : index    // Outer loop size (single iteration for simplicity)
    %II = arith.constant 4 : index   // Total tiles in I dimension
    %JJ = arith.constant 8 : index   // Total tiles in J dimension
    %NT_I = arith.constant 2 : index // Multi-tile factor I
    %NT_J = arith.constant 4 : index // Multi-tile factor J

    // Allocate 2D memref for library functions: [K, NT_I*NT_J]
    %load_memref_static = memref.alloca() : memref<1x8x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<1x8x!vx2> to memref<?x?x!vx2>

    // Loop over all tile indices like in GEMM (single k iteration)
    %global_stride_bytes = arith.constant 256 : index // SIZE_J * 2 = 128 * 2 bytes
    %elt_size_global = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Create conditional execution descriptor (k=0, cond_iter=0 to always execute when aligned)
        %cond_desc = aster_utils.struct_create(%c0, %c0, %NT_I, %NT_J) : (index, index, index, index) -> !conditional_execution_descriptor_2d

        // Call library function for global load
        // tensor_desc: m_pos=ii, n_pos=jj are tile indices
        %tensor_desc = aster_utils.struct_create(%in_ptr, %ii, %jj, %global_stride_bytes, %elt_size_global) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_maybe_global_load_multi_tile(
          %cond_desc,                   // conditional_execution_descriptor_2d
          %tensor_desc,                 // tensor_position_descriptor_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2d, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write
        %lds_stride_mt = arith.constant 256 : index // 128 * 2 bytes
        %elt_size_mt = arith.constant 2 : index
        %lds_pos_desc_mt = aster_utils.struct_create(%c0, %ii, %jj, %lds_stride_mt, %elt_size_mt) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        func.call @simple_maybe_lds_write_multi_tile(
          %cond_desc,                   // conditional_execution_descriptor_2d
          %lds_pos_desc_mt,             // lds_pos_desc_base (m_pos=ii, n_pos=jj as tile indices)
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant 256 : index // 128 * 2 bytes
    %elt_size_simple = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        %lds_pos_read = aster_utils.struct_create(%c0, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_simple) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %global_pos_write = aster_utils.struct_create(%out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_simple) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_read, %global_pos_write)
          : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
