// Minimal test for struct promotability via Mem2Reg and SROA
// Tests that simple struct types can be stored in memrefs and promoted
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!index_pair = !aster_utils.struct<i: index, j: index>

// CHECK-LABEL: amdgcn.module @test_struct_promotability_simple
amdgcn.module @test_struct_promotability_simple target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Test kernel that stores and loads a simple struct from memref
  // SROA and Mem2Reg should eliminate the memref operations
  // CHECK-LABEL: amdgcn.kernel @test_store_load_struct
  amdgcn.kernel @test_store_load_struct arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> {
    %out_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Allocate memref for struct (should be promoted by SROA + Mem2Reg)
    // CHECK-NOT: memref.alloca
    %struct_memref = memref.alloca() : memref<1x!index_pair>

    // Create struct and store in memref
    %struct = aster_utils.struct_create(%c1, %c2) : (index, index) -> !index_pair
    // CHECK-NOT: memref.store
    memref.store %struct, %struct_memref[%c0] : memref<1x!index_pair>

    // Load struct back from memref
    // CHECK-NOT: memref.load
    %loaded_struct = memref.load %struct_memref[%c0] : memref<1x!index_pair>

    // Extract fields and store to output
    // CHECK: aster_utils.struct_extract
    %i_val = aster_utils.struct_extract %loaded_struct["i"] : !index_pair -> index
    %j_val = aster_utils.struct_extract %loaded_struct["j"] : !index_pair -> index

    // Store i and j values to output buffer
    %i_i32 = arith.index_cast %i_val : index to i32
    %j_i32 = arith.index_cast %j_val : index to i32
    amdgcn.store global_store_dword data %i_i32 addr %out_ptr offset d(%c0) + c(%c0) : ins(i32, !amdgcn.sgpr_range<[? + 2]>, i32)
    %c4 = arith.constant 4 : index
    amdgcn.store global_store_dword data %j_i32 addr %out_ptr offset d(%c0) + c(%c4) : ins(i32, !amdgcn.sgpr_range<[? + 2]>, i32)

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
