// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

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
//   - cond_zero: condition index (execute only when cond_zero == 0)
//   - cond_mod_zero_i, cond_mod_zero_j: multi-tile factors (execute when ii % cond_mod_zero_i == 0 AND jj % cond_mod_zero_j == 0)
!conditional_execution_descriptor_2d = !aster_utils.struct<cond_zero: index, cond_mod_zero_i: index, cond_mod_zero_j: index>

amdgcn.module @test_lds_read_swizzled_A_fragment target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!tensor_position_descriptor_2d, !lds_position_descriptor_2d)
  func.func private @simple_global_store_wave_16x16xf16_wait(!vx2, !tensor_position_descriptor_2d)
  // From copies.mlir
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d) -> !vx2


  // Test @lds_read_swizzled_wave_16x16xf16_fragment_wait with XOR swizzling: read MFMA A fragment from LDS
  // Tests 2x3 tiles of 16x16, each tile contains iota 0-255
  // First populate LDS with known data, then read using the XOR-swizzled MFMA function
  amdgcn.kernel @test_lds_read_swizzled_A_wave_16x16xf16_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 4096 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %II = arith.constant 2 : index
    %JJ = arith.constant 3 : index

    %elt_size = arith.constant 2 : index
    // Global stride: JJ tiles * 16 elements * 2 bytes = JJ * 32 bytes
    %GLOBAL_STRIDE = affine.apply affine_map<()[JJ, elt_size] -> (JJ * 16 * elt_size)>()[%JJ, %elt_size]

    // LDS stride: same as global stride for this test
    %LDS_STRIDE = affine.apply affine_map<()[JJ, elt_size] -> (JJ * 16 * elt_size)>()[%JJ, %elt_size]

    // Load all 2x3 tiles from global to LDS
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %i_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %j_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]
        %global_pos_desc_load = aster_utils.struct_create(%in_ptr, %i_pos, %j_pos, %GLOBAL_STRIDE, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        %lds_pos_desc_load = aster_utils.struct_create(%c0, %i_pos, %j_pos, %LDS_STRIDE, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        func.call @simple_global_to_lds_wave_16x16xf16_wait(%global_pos_desc_load, %lds_pos_desc_load)
          : (!tensor_position_descriptor_2d, !lds_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read all tiles using XOR-swizzled read and store to output
    // Output layout: 6 tiles * 64 threads * 4 f16 = 1536 f16 values
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        %lds_pos_desc = aster_utils.struct_create(%c0, %m_pos, %n_pos, %LDS_STRIDE, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %fragment = func.call @lds_read_swizzled_wave_16x16xf16_fragment_wait(%lds_pos_desc)
          : (!lds_position_descriptor_2d) -> !vx2

        // Store fragment to output using simple_global_store_wave_16x16xf16_wait
        // Output buffer is treated as II*16 rows x JJ*16 columns with stride JJ*32 bytes
        %global_pos_desc_store = aster_utils.struct_create(%out_ptr, %m_pos, %n_pos, %GLOBAL_STRIDE, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_global_store_wave_16x16xf16_wait(%fragment, %global_pos_desc_store)
          : (!vx2, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

}
