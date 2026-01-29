// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!index_pair = !aster_utils.struct<i: index, j: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

amdgcn.module @test_global_load_ds_write target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  // From copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()


  // Test @global_load_wave_256xf16_via_dwordx2_wait + @lds_write_wave_256xf16_via_dwordx2_wait: decoupled global load and LDS write
  // Load from global to memref, then write from memref to LDS, then read back from LDS
  amdgcn.kernel @test_global_load_ds_write arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 512 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index // stride in bytes (16 elements * 2 bytes for f16)

    // Allocate memref for intermediate storage and cast to dynamic
    %memref_static = memref.alloca() : memref<1x1x!vx2>
    %memref = memref.cast %memref_static : memref<1x1x!vx2> to memref<?x?x!vx2>

    // Global load to memref, we know we are using 2B elements.
    %elt_size = arith.constant 2 : index
    %pos_desc = aster_utils.struct_create(%in_ptr, %c0, %c0, %c32, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_size_load = arith.constant 8 : index // dwordx2
    %wave_size_load = arith.constant 64 : index
    %transfer_desc_load = aster_utils.struct_create(%c16, %transfer_size_load, %wave_size_load) : (index, index, index) -> !transfer_descriptor_2d
    %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc, %transfer_desc_load) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)

    // DS write from memref to LDS
    %lds_pos_desc_write = aster_utils.struct_create(%c0, %c0, %c0, %c32, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
    %transfer_desc_write = aster_utils.struct_create(%c1, %transfer_size_load, %wave_size_load) : (index, index, index) -> !transfer_descriptor_2d
    func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc_write, %transfer_desc_write, %loaded)
      : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()

    // Read back from LDS and store to output
    %tid = gpu.thread_id x
    %lane = func.call @lane_id() : () -> index
    %dims = aster_utils.struct_create(%c16, %c4) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %iii, %jjj = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    %jjj_pos = affine.apply affine_map<()[jjj] -> (jjj * 4)>()[%jjj]

    %lds_off = affine.apply affine_map<()[iii, jjj_pos] -> ((iii * 16 + jjj_pos) * 2)>()[%iii, %jjj_pos]
    %lds_off_i32 = arith.index_cast %lds_off : index to i32
    %lds_off_vgpr = lsir.to_reg %lds_off_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %c0_i32 = arith.constant 0 : i32
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %lds_off_vgpr offset c(%c0_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    %tok_store = amdgcn.store global_store_dwordx2 data %from_lds addr %out_ptr offset d(%out_off_vgpr) + c(%c0_i32) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

}
