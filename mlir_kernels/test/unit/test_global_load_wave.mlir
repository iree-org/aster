// Unit test kernels for global_load_wave library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx3 = !amdgcn.vgpr<[? + 3]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // From copies.mlir
  func.func private @global_load_wave_128xf16_via_dword_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx1)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @global_load_wave_384xf16_via_dwordx3_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx3)
  func.func private @global_load_wave_512xf16_via_dwordx4_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx4)

  func.func private @get_test_offset(%transfer_size: index) -> (!v) {
    %tid = gpu.thread_id x
    %offset = affine.apply affine_map<()[tid, transfer_size]
      -> (tid * transfer_size)>()[%tid, %transfer_size]
    %offset_i32 = arith.index_cast %offset : index to i32
    %offset_vgpr = lsir.to_reg %offset_i32 : i32 -> !v
    return %offset_vgpr : !v
  }

  // Load from global to registers, then write to global.
  amdgcn.kernel @test_global_load_wave arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr_vx1 = amdgcn.load_arg 1 : !sx2
    %out_ptr_vx2 = amdgcn.load_arg 2 : !sx2
    %out_ptr_vx3 = amdgcn.load_arg 3 : !sx2
    %out_ptr_vx4 = amdgcn.load_arg 4 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    //===--------------------------------------------------------------------===//
    // dword
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx1 = arith.constant 4 : index
    %elt_size = arith.constant 2 : index
    %wave_size = arith.constant 64 : index
    %pos_desc_vx1 = aster_utils.struct_create(%in_ptr, %c0, %c0, %c0, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_desc_vx1 = aster_utils.struct_create(%c1, %transfer_size_vx1, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
    %loaded_vx1 = func.call @global_load_wave_128xf16_via_dword_wait(%pos_desc_vx1, %transfer_desc_vx1) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx1)
    %out_off_vx1 = func.call @get_test_offset(%transfer_size_vx1) : (index) -> (!v)
    %tok_store_1 = amdgcn.store global_store_dword data %loaded_vx1 addr %out_ptr_vx1 offset d(%out_off_vx1)
      : ins(!vx1, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx2
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx2 = arith.constant 8 : index
    %pos_desc_vx2 = aster_utils.struct_create(%in_ptr, %c0, %c0, %c0, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_desc_vx2 = aster_utils.struct_create(%c1, %transfer_size_vx2, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
    %loaded_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc_vx2, %transfer_desc_vx2) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
    %out_off_vx2 = func.call @get_test_offset(%transfer_size_vx2) : (index) -> (!v)
    %tok_store_2 = amdgcn.store global_store_dwordx2 data %loaded_vx2 addr %out_ptr_vx2 offset d(%out_off_vx2)
      : ins(!vx2, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx3
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx3 = arith.constant 12 : index
    %pos_desc_vx3 = aster_utils.struct_create(%in_ptr, %c0, %c0, %c0, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_desc_vx3 = aster_utils.struct_create(%c1, %transfer_size_vx3, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
    %loaded_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_wait(%pos_desc_vx3, %transfer_desc_vx3) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx3)
    %out_off_vx3 = func.call @get_test_offset(%transfer_size_vx3) : (index) -> (!v)
    %tok_store_3 = amdgcn.store global_store_dwordx3 data %loaded_vx3 addr %out_ptr_vx3 offset d(%out_off_vx3)
      : ins(!vx3, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx4
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx4 = arith.constant 16 : index
    %pos_desc_vx4 = aster_utils.struct_create(%in_ptr, %c0, %c0, %c0, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_desc_vx4 = aster_utils.struct_create(%c1, %transfer_size_vx4, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
    %loaded_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_wait(%pos_desc_vx4, %transfer_desc_vx4) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx4)
    %out_off_vx4 = func.call @get_test_offset(%transfer_size_vx4) : (index) -> (!v)
    %tok_store_4 = amdgcn.store global_store_dwordx4 data %loaded_vx4 addr %out_ptr_vx4 offset d(%out_off_vx4)
      : ins(!vx4, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

}
