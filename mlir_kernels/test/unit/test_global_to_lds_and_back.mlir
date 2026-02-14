// Unit test for @simple_global_to_lds_wave_16x16xf16_wait and @simple_lds_to_global_wave_16x16xf16_wait
// Copy a single 16x16 tile from global to LDS and back

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

amdgcn.module @test_global_to_lds_and_back target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!tensor_position_descriptor_2d, !lds_position_descriptor_2d)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(!lds_position_descriptor_2d, !tensor_position_descriptor_2d)

  // Test @simple_global_to_lds_wave_16x16xf16_wait: copy a single 16x16 tile from global to LDS
  // Input: 64x96 array, copy tile at position (3,5) = element (48, 80)
  // Verifies position handling by checking only the correct tile is copied
  amdgcn.kernel @test_global_to_lds_and_back_wave_16x16xf16_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 20000 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index   // m_pos = 1 * 16
    %c32 = arith.constant 32 : index   // n_pos = 2 * 16
    %c120 = arith.constant 120 : index // global stride = 60 * 2 bytes
    %c32_2 = arith.constant 32 : index   // LDS stride = 16 * 2 bytes
    %elt_size = arith.constant 2 : index // f16 size in bytes

    // Copy tile at (48, 80) from global to LDS at base 0
    %global_pos_desc = aster_utils.struct_create(%in_ptr, %c16, %c32, %c120, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %lds_pos_desc = aster_utils.struct_create(%c0, %c16, %c32, %c32_2, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
    func.call @simple_global_to_lds_wave_16x16xf16_wait(%global_pos_desc, %lds_pos_desc)
      : (!tensor_position_descriptor_2d, !lds_position_descriptor_2d) -> ()

    // Copy from LDS to global at position (48, 80)
    %global_pos_desc_out = aster_utils.struct_create(%out_ptr, %c16, %c32, %c120, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_desc, %global_pos_desc_out)
      : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
