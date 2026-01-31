// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // From register-init.mlir
  // From indexing.mlir
  // From simple-copies.mlir
  // From copies.mlir
  func.func private @store_to_global_dword_wait(!v, !tensor_position_descriptor_2d)
  func.func private @store_to_global_dwordx2_wait(!vx2, !tensor_position_descriptor_2d)
  func.func private @store_to_global_dwordx3_wait(!vx3, !tensor_position_descriptor_2d)
  func.func private @store_to_global_dwordx4_wait(!vx4, !tensor_position_descriptor_2d)

  //===--------------------------------------------------------------------===//
  // Global store
  //===--------------------------------------------------------------------===//

  // Test @store_to_global_dword_wait: store a dword to global memory
  // Each thread stores (tid * 100) at position (tid/8, tid%8) in a 16-wide matrix
  amdgcn.kernel @test_store_to_global_dword_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)

    // Compute i, j from tid
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]

    // Compute value to store: tid * 100
    %value_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %value_i32 = arith.index_cast %value_idx : index to i32
    %value = lsir.to_reg %value_i32 : i32 -> !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    %elt_size = arith.constant 4 : index  // dword = 4 bytes
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %tid, %c0, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dword_wait(%value, %pos_desc)
      : (!v, !tensor_position_descriptor_2d) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx2_wait: store 8 bytes to global memory
  // Each thread stores 2 dwords at position (tid/4, tid%4) in an 8-wide matrix
  amdgcn.kernel @test_store_to_global_dwordx2_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index // stride in bytes (8 elements * 8 bytes)

    // Compute i, j from tid (each thread writes 2 dwords = 1 column)
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 4)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 4)>()[%tid]

    // Compute values to store: [tid * 100, tid * 100 + 1]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1 : !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    %elt_size = arith.constant 8 : index  // dwordx2 = 8 bytes
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %tid, %c0, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dwordx2_wait(%value, %pos_desc)
      : (!vx2, !tensor_position_descriptor_2d) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx3_wait: store 12 bytes to global memory
  // Each thread stores 3 dwords at linear position tid
  amdgcn.kernel @test_store_to_global_dwordx3_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x

    // Compute values to store: [tid * 100, tid * 100 + 1, tid * 100 + 2]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v2_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 2)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v2_i32 = arith.index_cast %v2_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %v2 = lsir.to_reg %v2_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1, %v2 : !v, !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    %elt_size = arith.constant 12 : index  // dwordx3 = 12 bytes
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %tid, %c0, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dwordx3_wait(%value, %pos_desc)
      : (!vx3, !tensor_position_descriptor_2d) -> ()

    amdgcn.end_kernel
  }

  // Test @store_to_global_dwordx4_wait: store 16 bytes to global memory
  // Each thread stores 4 dwords at position (tid/4, tid%4) in a 4-wide matrix
  amdgcn.kernel @test_store_to_global_dwordx4_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x

    // Compute values to store: [tid * 100, tid * 100 + 1, tid * 100 + 2, tid * 100 + 3]
    %v0_idx = affine.apply affine_map<()[tid] -> (tid * 100)>()[%tid]
    %v1_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 1)>()[%tid]
    %v2_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 2)>()[%tid]
    %v3_idx = affine.apply affine_map<()[tid] -> (tid * 100 + 3)>()[%tid]
    %v0_i32 = arith.index_cast %v0_idx : index to i32
    %v1_i32 = arith.index_cast %v1_idx : index to i32
    %v2_i32 = arith.index_cast %v2_idx : index to i32
    %v3_i32 = arith.index_cast %v3_idx : index to i32
    %v0 = lsir.to_reg %v0_i32 : i32 -> !v
    %v1 = lsir.to_reg %v1_i32 : i32 -> !v
    %v2 = lsir.to_reg %v2_i32 : i32 -> !v
    %v3 = lsir.to_reg %v3_i32 : i32 -> !v
    %value = amdgcn.make_register_range %v0, %v1, %v2, %v3 : !v, !v, !v, !v

    // Store using the library function (single row, stride must not matter)
    %c0 = arith.constant 0 : index
    %elt_size = arith.constant 16 : index  // dwordx4 = 16 bytes
    %pos_desc = aster_utils.struct_create(%out_ptr, %c0, %tid, %c0, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    func.call @store_to_global_dwordx4_wait(%value, %pos_desc)
      : (!vx4, !tensor_position_descriptor_2d) -> ()

    amdgcn.end_kernel
  }

}
