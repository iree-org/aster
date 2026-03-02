// Test kernel for kittens @zero_C function.
// Verifies that zero initialization produces correct output.

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_C_f32 = !vx4
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

amdgcn.module @test_kittens_zero_C target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  // From copies.mlir
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !tensor_position_descriptor_2level_2d, i1)

  // Test @zero_C: initialize accumulator to zero and store to global
  amdgcn.kernel @test_zero_C arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index  // stride in bytes (16 elements * 4 bytes for f32)
    %elt_size = arith.constant 4 : index

    // Initialize accumulator to zero using kittens function
    %acc = func.call @zero_C() : () -> !rt_C_f32

    // Store using the library function
    // m_pos=0, n_pos=0, mm_pos=0, nn_pos=0
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %c0, %c64, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %false = arith.constant false
    func.call @global_store_wave_16x16xf32_C_fragment_wait(%acc, %pos_desc, %false) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

    amdgcn.end_kernel
  }
}
