// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

amdgcn.module @test_load_and_lds_read_A_fragment_transposed target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(!tensor_position_descriptor_2level_2d, index, index)
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d, i1) -> !vx2


  // Test @test_load_and_lds_read_A_wave_16x16xf16_fragment_transposed_wait: read MFMA A fragment from LDS
  // First populate LDS with known data, then read using the MFMA function
  amdgcn.kernel @test_load_and_lds_read_A_wave_16x16xf16_fragment_transposed_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 512 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index // stride in bytes (16 elements * 2 bytes for f16)

    // First load data to LDS using load_to_lds
    %elt_size = arith.constant 2 : index
    %pos_desc = aster_utils.struct_create(%in_ptr, %c0, %c0, %c32, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    func.call @global_load_to_lds_wave_16x16_f16_wait(%pos_desc, %c0, %c32) : (!tensor_position_descriptor_2level_2d, index, index) -> ()

    // Now read the A fragment using the MFMA read
    // i_pos=0, j_pos=0
    %true = arith.constant true
    %lds_pos_desc = aster_utils.struct_create(%c0, %c0, %c0, %c32, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
    %fragment = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc, %true)
      : (!lds_position_descriptor_2d, i1) -> !vx2

    // Store fragment to output (each thread writes 8 bytes)
    %tid = gpu.thread_id x
    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    %c0_store = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dwordx2 data %fragment addr %out_ptr offset d(%out_off_vgpr) + c(%c0_store) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

}
