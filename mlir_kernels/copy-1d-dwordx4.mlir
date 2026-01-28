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
!tensor_position_descriptor_1d = !aster_utils.struct<ptr: !sx2, pos: index, stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!m2reg_param_1d_vx4 = !aster_utils.struct<i: index, memref: memref<?x!vx4>>

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From indexing.mlir
  func.func private @distributed_index_1d() -> index
  func.func private @grid_stride_1d() -> index
  // From copies.mlir
  func.func private @store_to_global_dwordx4_wait(!vx4, !tensor_position_descriptor_2d) -> ()
  func.func private @load_from_global_dwordx4_wait(!tensor_position_descriptor_2d) -> !vx4

  // Global load helper - loads dwordx4 at element position
  // Note: This is a global adapter that converts a 1D position descriptor to a
  // 2D position descriptor for the load_from_global_dwordx4_wait API. This can
  // be moved to copies.mlir or to a copies_adapter.mlir once we settle on the
  // right dimensions for the API and how canonicalizations and foldings work at
  // scale.
  func.func private @global_load_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {
    // Extract index and memref from mem2reg parameter
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>

    // Convert 1D to 2D descriptor (m_pos=0, n_pos=pos) for the load_from_global_dwordx4_wait API.
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Use library function to load dwordx4
    %loaded = func.call @load_from_global_dwordx4_wait(%pos_desc_2d) : (!tensor_position_descriptor_2d) -> !vx4

    // Store to memref for later use
    memref.store %loaded, %memref[%idx] : memref<?x!vx4>
    return
  }

  // Global store helper - stores dwordx4 at element position
  // Note: This is a global adapter that converts a 1D position descriptor to a
  // 2D position descriptor for the store_to_global_dwordx4_wait API. This can
  // be moved to copies.mlir or to a copies_adapter.mlir once we settle on the
  // right dimensions for the API and how canonicalizations and foldings work at
  // scale.
  func.func private @global_store_body(
    %pos_desc_1d: !tensor_position_descriptor_1d,
    %m2reg_param: !m2reg_param_1d_vx4
  ) {
    // Extract index and memref from mem2reg parameter
    %idx, %memref = aster_utils.struct_extract %m2reg_param["i", "memref"] : !m2reg_param_1d_vx4 -> index, memref<?x!vx4>

    // Load from memref (for SROA + MEM2REG)
    %value = memref.load %memref[%idx] : memref<?x!vx4>

    // Convert 1D to 2D descriptor (m_pos=0, n_pos=pos) for the store_to_global_dwordx4_wait API.
    %c0 = arith.constant 0 : index
    %ptr, %pos, %stride, %elt_size = aster_utils.struct_extract %pos_desc_1d["ptr", "pos", "stride_in_bytes", "elt_size"] : !tensor_position_descriptor_1d -> !sx2, index, index, index
    %pos_desc_2d = aster_utils.struct_create(%ptr, %c0, %pos, %stride, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Use library function to store dwordx4
    func.call @store_to_global_dwordx4_wait(%value, %pos_desc_2d)
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
    %c16 = arith.constant 16 : index  // dwordx4 size in bytes

    // Get base distributed index and grid stride for grid-stride loop pattern
    %base_index = func.call @distributed_index_1d() : () -> index
    %grid_stride = func.call @grid_stride_1d() : () -> index
    // Convert grid stride to byte stride (grid_stride * element_size)
    %grid_stride_bytes = affine.apply affine_map<(s)[elt] -> (s * elt)>(%grid_stride)[%c16]

    // Loop over elements per thread
    scf.for %i = %c0 to %num_elements step %c1 {
      // Calculate strided element index: base_index + i * grid_stride
      %elem_index = affine.apply affine_map<(base, i)[stride] -> (base + i * stride)>
        (%base_index, %i)[%grid_stride]

      // Create 1D position descriptors
      %src_pos_desc = aster_utils.struct_create(%src_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d
      %dst_pos_desc = aster_utils.struct_create(%dst_global, %elem_index, %grid_stride_bytes, %c16) : (!sx2, index, index, index) -> !tensor_position_descriptor_1d

      // Create mem2reg parameter descriptor
      %m2reg_param = aster_utils.struct_create(%i, %memref) : (index, memref<?x!vx4>) -> !m2reg_param_1d_vx4

      func.call @global_load_body(%src_pos_desc, %m2reg_param)
        {sched.delay = 0 : i32, sched.rate = 1 : i32}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
      func.call @global_store_body(%dst_pos_desc, %m2reg_param)
        {sched.delay = {{SCHED_DELAY_STORE}} : i32, sched.rate = 1 : i32}
        : (!tensor_position_descriptor_1d, !m2reg_param_1d_vx4) -> ()
    } {sched.dims = array<i64: {{NUM_ELEMENTS_PER_THREAD}}>}

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

    %num_elements = arith.constant {{NUM_ELEMENTS_PER_THREAD}} : index
    func.call @copy_loop(%num_elements, %src_ptr, %dst_ptr)
      : (index, !sx2, !sx2) -> ()

    amdgcn.end_kernel
  }
}
