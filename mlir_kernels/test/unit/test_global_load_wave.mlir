// Unit test kernels for global_load_wave library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr_range<[? + 1]>
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx3 = !amdgcn.sgpr_range<[? + 3]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // copies.mlir
  func.func private @global_load_wave_128xf16_via_dword_wait(!sx2, index, index, index, index, index, index) -> (!vx1)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @global_load_wave_384xf16_via_dwordx3_wait(!sx2, index, index, index, index, index, index) -> (!vx3)
  func.func private @global_load_wave_512xf16_via_dwordx4_wait(!sx2, index, index, index, index, index, index) -> (!vx4)

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
    %loaded_vx1 = func.call @global_load_wave_128xf16_via_dword_wait(
      %in_ptr,   // ptr
      %c0, %c0,  // m_pos, n_pos (major tile)
      %c0,       // GLOBAL_STRIDE_IN_BYTES (single row, stride must not matter)
      %c0, %c0,  // mm_pos, nn_pos (minor tile)
      %c1        // num_rows
    ) : (!sx2, index, index, index, index, index, index) -> (!vx1)
    %out_off_vx1 = func.call @get_test_offset(%transfer_size_vx1) : (index) -> (!v)
    %tok_store_1 = amdgcn.store global_store_dword data %loaded_vx1 addr %out_ptr_vx1 offset d(%out_off_vx1)
      : ins(!vx1, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx2
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx2 = arith.constant 8 : index
    %loaded_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_wait(
      %in_ptr,   // ptr
      %c0, %c0,  // m_pos, n_pos (major tile)
      %c0,       // GLOBAL_STRIDE_IN_BYTES (single row, stride must not matter)
      %c0, %c0,  // mm_pos, nn_pos (minor tile)
      %c1        // num_rows
    ) : (!sx2, index, index, index, index, index, index) -> (!vx2)
    %out_off_vx2 = func.call @get_test_offset(%transfer_size_vx2) : (index) -> (!v)
    %tok_store_2 = amdgcn.store global_store_dwordx2 data %loaded_vx2 addr %out_ptr_vx2 offset d(%out_off_vx2)
      : ins(!vx2, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx3
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx3 = arith.constant 12 : index
    %loaded_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_wait(
      %in_ptr,   // ptr
      %c0, %c0,  // m_pos, n_pos (major tile)
      %c0,       // GLOBAL_STRIDE_IN_BYTES (single row, stride must not matter)
      %c0, %c0,  // mm_pos, nn_pos (minor tile)
      %c1        // num_rows
    ) : (!sx2, index, index, index, index, index, index) -> (!vx3)
    %out_off_vx3 = func.call @get_test_offset(%transfer_size_vx3) : (index) -> (!v)
    %tok_store_3 = amdgcn.store global_store_dwordx3 data %loaded_vx3 addr %out_ptr_vx3 offset d(%out_off_vx3)
      : ins(!vx3, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    //===--------------------------------------------------------------------===//
    // dwordx4
    //===--------------------------------------------------------------------===//
    // Global load to registers.
    %transfer_size_vx4 = arith.constant 16 : index
    %loaded_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_wait(
      %in_ptr,   // ptr
      %c0, %c0,  // m_pos, n_pos (major tile)
      %c0,       // GLOBAL_STRIDE_IN_BYTES (single row, stride must not matter)
      %c0, %c0,  // mm_pos, nn_pos (minor tile)
      %c1        // num_rows
    ) : (!sx2, index, index, index, index, index, index) -> (!vx4)
    %out_off_vx4 = func.call @get_test_offset(%transfer_size_vx4) : (index) -> (!v)
    %tok_store_4 = amdgcn.store global_store_dwordx4 data %loaded_vx4 addr %out_ptr_vx4 offset d(%out_off_vx4)
      : ins(!vx4, !sx2, !v) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

}
