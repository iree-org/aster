// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{GRID_DIM_X}}/1/g' -e 's/{{BLOCK_DIM_X}}/64/g' -e 's/{{NUM_ELEMENTS_PER_THREAD}}/1/g' -e 's/{{SCHED_DELAY_STORE}}/3/g' \
// RUN: | aster-opt \
// RUN: | FileCheck %s

// Minimal 1-D copy kernel using dwordx4 (16 bytes per thread)

// CHECK-LABEL: amdgcn.module

amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // Global load helper - loads dwordx4 at iteration index
  func.func private @global_load_body(
    %threadidx_x: index,
    %blockidx_x: index,
    %blockdim_x: index,
    %griddim_x: index,
    %iter: index,
    %src_global: !amdgcn.sgpr_range<[? + 2]>,
    %memref: memref<?x!amdgcn.vgpr_range<[? + 4]>>
  ) {
    // Allocate registers for dwordx4
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr1 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr2 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr3 = amdgcn.alloca : !amdgcn.vgpr
    %range = amdgcn.make_register_range %vgpr0, %vgpr1, %vgpr2, %vgpr3
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr

    // Calculate byte offset: ((iter * griddim * blockdim) + (blockidx * blockdim) + threadidx) * 16
    %offset_index = affine.apply affine_map<
      (iter, bidx, tidx)[gdim, bdim] -> (((iter * gdim + bidx) * bdim + tidx) * 16)>
      (%iter, %blockidx_x, %threadidx_x)[%griddim_x, %blockdim_x]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // Global load dwordx4
    %loaded = amdgcn.flat.global_load #amdgcn.inst<global_load_dwordx4> %range, %src_global[%offset_vgpr]
      : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr] -> !amdgcn.vgpr_range<[? + 4]>

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Store to memref for later use
    memref.store %loaded, %memref[%iter] : memref<?x!amdgcn.vgpr_range<[? + 4]>>
    return
  }

  // Global store helper - stores dwordx4 at iteration index
  func.func private @global_store_body(
    %threadidx_x: index,
    %blockidx_x: index,
    %blockdim_x: index,
    %griddim_x: index,
    %iter: index,
    %memref: memref<?x!amdgcn.vgpr_range<[? + 4]>>,
    %dst_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    // Load from memref (for SROA + MEM2REG)
    %value = memref.load %memref[%iter] : memref<?x!amdgcn.vgpr_range<[? + 4]>>

    // Calculate byte offset: ((iter * griddim * blockdim) + (blockidx * blockdim) + threadidx) * 16
    %offset_index = affine.apply affine_map<
      (iter, bidx, tidx)[gdim, bdim] -> (((iter * gdim + bidx) * bdim + tidx) * 16)>
      (%iter, %blockidx_x, %threadidx_x)[%griddim_x, %blockdim_x]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // Global store dwordx4
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx4> %value, %dst_global[%offset_vgpr]
      : !amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>[!amdgcn.vgpr]

    // Wait for store completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // Main copy loop function
  func.func private @copy_loop(
    %num_elements: index,
    %src_global: !amdgcn.sgpr_range<[? + 2]>,
    %dst_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    // Allocate memref for data transfer between load and store
    %memref = memref.alloca(%num_elements) : memref<?x!amdgcn.vgpr_range<[? + 4]>>

    // Get thread/block indices
    %threadidx_x = gpu.thread_id x
    %blockidx_x = gpu.block_id x
    %blockdim_x = gpu.block_dim x
    %griddim_x = gpu.grid_dim x

    // Loop constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Loop over elements per thread
    scf.for %i = %c0 to %num_elements step %c1 {
      func.call @global_load_body(%threadidx_x, %blockidx_x, %blockdim_x, %griddim_x, %i, %src_global, %memref)
        {sched.delay = 0 : i32, sched.rate = 1 : i32}
        : (index, index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x!amdgcn.vgpr_range<[? + 4]>>) -> ()
      func.call @global_store_body(%threadidx_x, %blockidx_x, %blockdim_x, %griddim_x, %i, %memref, %dst_global)
        {sched.delay = {{SCHED_DELAY_STORE}} : i32, sched.rate = 1 : i32}
        : (index, index, index, index, index, memref<?x!amdgcn.vgpr_range<[? + 4]>>, !amdgcn.sgpr_range<[? + 2]>) -> ()
    } {sched.dims = array<i64: {{NUM_ELEMENTS_PER_THREAD}}>}

    return
  }

  // Test function that calls copy_loop with templated dimensions
  func.func private @test_copy(
    %src_global: !amdgcn.sgpr_range<[? + 2]>,
    %dst_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    %num_elements = arith.constant {{NUM_ELEMENTS_PER_THREAD}} : index

    func.call @copy_loop(%num_elements, %src_global, %dst_global)
      : (index, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    return
  }

  amdgcn.kernel @copy_1d_dwordx4_static arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    // Load kernel arguments
    %src_ptr_s = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %dst_ptr_s = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %src_ptr, %dst_ptr = lsir.assume_noalias %src_ptr_s, %dst_ptr_s
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    func.call @test_copy(%src_ptr, %dst_ptr)
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    amdgcn.end_kernel
  }
}
