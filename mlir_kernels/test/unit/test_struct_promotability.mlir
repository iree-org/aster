// Minimal test for struct promotability via Mem2Reg and SROA
// Tests that struct types (futures) can be stored in memrefs and promoted
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// CHECK-LABEL: amdgcn.module @test_struct_promotability
amdgcn.module @test_struct_promotability target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From copies.mlir - function that returns a future
  func.func private @load_from_global_dwordx2_future(!tensor_position_descriptor_2d) -> !future_global_read_any

  // Test kernel that stores and loads a struct from memref
  // SROA and Mem2Reg should eliminate the memref operations
  // CHECK-LABEL: amdgcn.kernel @test_store_load_future
  amdgcn.kernel @test_store_load_future arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %elt_size = arith.constant 2 : index // f16 size in bytes

    // Create a simple position descriptor
    %m_pos = arith.constant 0 : index
    %n_pos = arith.constant 0 : index
    %stride = arith.constant 256 : index
    %pos_desc = aster_utils.struct_create(%in_ptr, %m_pos, %n_pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Allocate memref for future (should be promoted by SROA + Mem2Reg)
    // CHECK-NOT: memref.alloca
    %future_memref = memref.alloca() : memref<1x!future_global_read_any>

    // Load future and store in memref
    %future = func.call @load_from_global_dwordx2_future(%pos_desc) : (!tensor_position_descriptor_2d) -> !future_global_read_any
    // CHECK-NOT: memref.store
    memref.store %future, %future_memref[%c0] : memref<1x!future_global_read_any>

    // Load future back from memref
    // CHECK-NOT: memref.load
    %loaded_future = memref.load %future_memref[%c0] : memref<1x!future_global_read_any>

    // Extract value and token from future
    // CHECK: aster_utils.struct_extract
    %value_any, %token = aster_utils.struct_extract %loaded_future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>

    // Wait on token
    amdgcn.wait deps %token : !amdgcn.read_token<flat>

    // Convert value back and store to output
    %value = aster_utils.from_any %value_any : !vx2
    amdgcn.store global_store_dwordx2 data %value addr %out_ptr offset d(%c0) + c(%c0) : ins(!vx2, !sx2, i32)

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
