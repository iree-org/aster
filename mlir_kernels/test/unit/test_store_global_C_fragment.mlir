// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr<[? + 4]>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

amdgcn.module @test_store_global_C_fragment target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From register-init.mlir
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // From indexing.mlir
  func.func private @lane_id() -> index
  // From copies.mlir
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !tensor_position_descriptor_2level_2d, i1)


  // Test @global_store_wave_16x16xf32_C_fragment_wait: store C fragment to global
  // Initialize accumulators with known values, then store using MFMA function
  amdgcn.kernel @test_store_global_C_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes for f32)

    // Initialize accumulator with lane_id (as float bits)
    %lane = func.call @lane_id() : () -> index
    %lane_i32 = arith.index_cast %lane : index to i32
    %lane_reg = lsir.to_reg %lane_i32 : i32 -> !v
    %acc = func.call @init_vgprx4_reg(%lane_reg) : (!v) -> !vx4

    // Store using the library function
    // i_pos=0, j_pos=0, ii_pos=0, jj_pos=0
    %elt_size = arith.constant 4 : index
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %c0, %c64, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %false = arith.constant false
    func.call @global_store_wave_16x16xf32_C_fragment_wait(%acc, %pos_desc, %false) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

    amdgcn.end_kernel
  }

}
