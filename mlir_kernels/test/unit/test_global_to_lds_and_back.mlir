// Unit test for @simple_global_to_lds_wave_16x16xf16_wait and @simple_lds_to_global_wave_16x16xf16_wait
// Copy a single 16x16 tile from global to LDS and back

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

amdgcn.module @test_global_to_lds_and_back target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
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
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, index, index, memref<?x?x!vx2>)
  // From simple-multi-tile-copies.mlir
  func.func private @simple_maybe_lds_write_multi_tile(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)
  // From multi-tile-copies.mlir
  func.func private @simple_maybe_global_load_multi_tile(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2d, memref<?x?x!vx2>)
  func.func private @maybe_global_load_multi_tile_coalesced(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)

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
