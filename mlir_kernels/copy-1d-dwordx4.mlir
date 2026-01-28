// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{GRID_DIM_X}}/1/g' -e 's/{{BLOCK_DIM_X}}/64/g' -e 's/{{NUM_ELEMENTS_PER_THREAD}}/1/g' -e 's/{{SCHED_DELAY_STORE}}/3/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/copies.mlir,%p/library/common/indexing.mlir,%p/library/common/register-init.mlir" \
// RUN: | FileCheck %s

// Minimal 1-D copy kernel using dwordx4 (16 bytes per thread)

// CHECK-LABEL: amdgcn.module

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From indexing.mlir
  func.func private @distributed_index_1d_with_grid_stride(index) -> index
  // From copies.mlir
  func.func private @store_to_global_dwordx4_wait(!vx4, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dwordx4_wait(!tensor_position_descriptor_2d) -> !vx4

  // Global load helper - loads dwordx4 at element position
  func.func private @global_load_body(
    %elem_index: index,
    %iter: index,
    %src_global: !sx2,
    %memref: memref<?x!vx4>
  ) {
    // Create position descriptor for 1D access (treat as 2D with m_pos=0)
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index  // dwordx4 size in bytes
    %pos_desc = aster_utils.struct_create(%src_global, %c0, %elem_index, %c16, %c16) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Use library function to load dwordx4
    %loaded = func.call @load_from_global_dwordx4_wait(%pos_desc) : (!tensor_position_descriptor_2d) -> !vx4

    // Store to memref for later use
    memref.store %loaded, %memref[%iter] : memref<?x!vx4>
    return
  }

  // Global store helper - stores dwordx4 at element position
  func.func private @global_store_body(
    %elem_index: index,
    %iter: index,
    %memref: memref<?x!vx4>,
    %dst_global: !sx2
  ) {
    // Load from memref (for SROA + MEM2REG)
    %value = memref.load %memref[%iter] : memref<?x!vx4>

    // Create position descriptor for 1D access (treat as 2D with m_pos=0)
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index  // dwordx4 size in bytes
    %pos_desc = aster_utils.struct_create(%dst_global, %c0, %elem_index, %c16, %c16) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Use library function to store dwordx4
    func.call @store_to_global_dwordx4_wait(%value, %pos_desc)
      : (!vx4, !tensor_position_descriptor_2d) -> ()
    return
  }

  // Main copy loop function
  func.func private @copy_loop(
    %num_elements: index,
    %src_global: !sx2,
    %dst_global: !sx2
  ) {
    // Allocate memref for data transfer between load and store
    %memref = memref.alloca(%num_elements) : memref<?x!vx4>

    // Loop constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Loop over elements per thread
    scf.for %i = %c0 to %num_elements step %c1 {
      // Calculate element index using grid-stride pattern (internally queries GPU intrinsics)
      %elem_index = func.call @distributed_index_1d_with_grid_stride(%i) : (index) -> index

      func.call @global_load_body(%elem_index, %i, %src_global, %memref)
        {sched.delay = 0 : i32, sched.rate = 1 : i32}
        : (index, index, !sx2, memref<?x!vx4>) -> ()
      func.call @global_store_body(%elem_index, %i, %memref, %dst_global)
        {sched.delay = {{SCHED_DELAY_STORE}} : i32, sched.rate = 1 : i32}
        : (index, index, memref<?x!vx4>, !sx2) -> ()
    } {sched.dims = array<i64: {{NUM_ELEMENTS_PER_THREAD}}>}

    return
  }

  // Test function that calls copy_loop with templated dimensions
  func.func private @test_copy(
    %src_global: !sx2,
    %dst_global: !sx2
  ) {
    %num_elements = arith.constant {{NUM_ELEMENTS_PER_THREAD}} : index

    func.call @copy_loop(%num_elements, %src_global, %dst_global)
      : (index, !sx2, !sx2) -> ()

    return
  }

  amdgcn.kernel @copy_1d_dwordx4_static arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    // Load kernel arguments
    %src_ptr_s = amdgcn.load_arg 0 : !sx2
    %dst_ptr_s = amdgcn.load_arg 1 : !sx2
    %src_ptr, %dst_ptr = lsir.assume_noalias %src_ptr_s, %dst_ptr_s
      : (!sx2, !sx2)
      -> (!sx2, !sx2)
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    func.call @test_copy(%src_ptr, %dst_ptr)
      : (!sx2, !sx2) -> ()

    amdgcn.end_kernel
  }
}
