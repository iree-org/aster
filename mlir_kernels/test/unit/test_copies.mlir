// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register_init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @swizzle_A_16x16xf16() -> (index, index)
  func.func private @swizzle_C_16x16xf32() -> (index, index)
  // copies.mlir
  func.func private @load_to_lds_16x16_dwordx2_wait(!sx2, index, index, index, index, index, index, index)
  func.func private @global_load_dwordx2_wait(!sx2, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)
  func.func private @ds_write_dwordx2_wait(index, index, index, index, index, index, index, memref<?x?x!vx2>)
  func.func private @store_to_global_dword_wait(!v, !sx2, index, index, index)
  func.func private @read_lds_A_16x16xf16_fragment_wait(index, index, index, index) -> !vx2
  func.func private @store_global_16x16xf32_C_fragment_wait(!vx4, !sx2, index, index, index, index, index)

  //===--------------------------------------------------------------------===//
  // Helper: store i32 to global at thread index
  //===--------------------------------------------------------------------===//
  func.func private @store_at_tid(%value: i32, %ptr: !sx2, %index_offset: index) {
    %tid = gpu.thread_id x
    %value_vgpr = lsir.to_reg %value : i32 -> !v

    %offset_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 4 + index_offset)>()[%tid, %index_offset]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !v

    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %value_vgpr, %ptr[%offset_vgpr]
      : !v, !sx2[!v]

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return
  }

  //===--------------------------------------------------------------------===//
  // Test kernels - one per function
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
    %c16 = arith.constant 16 : index

    // Compute i, j from tid
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]

    // Compute value to store: tid * 100
    %tid_i32 = arith.index_cast %tid : index to i32
    %c100 = arith.constant 100 : i32
    %value_i32 = arith.muli %tid_i32, %c100 : i32
    %value = lsir.to_reg %value_i32 : i32 -> !v

    // Store using the library function
    func.call @store_to_global_dword_wait(%value, %out_ptr, %i, %j, %c16)
      : (!v, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test @read_lds_A_16x16xf16_fragment_wait: read swizzled A fragment from LDS
  // First populate LDS with known data, then read using the swizzled function
  amdgcn.kernel @test_load_and_read_lds_A_16x16xf16_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 512 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index

    // First load data to LDS using load_to_lds
    func.call @load_to_lds_16x16_dwordx2_wait(
      %in_ptr, %c0,      // ptr, lds_base_off
      %c0, %c0,          // i_pos, j_pos
      %c16,              // N_SIZE
      %c0, %c0,          // ii_pos, jj_pos
      %c16               // NN_SIZE
    ) : (!sx2, index, index, index, index, index, index, index) -> ()

    // Now read the A fragment using the swizzled read
    // i_pos=0, j_pos=0, N_SIZE=16
    %fragment = func.call @read_lds_A_16x16xf16_fragment_wait(%c0, %c0, %c0, %c16)
      : (index, index, index, index) -> !vx2

    // Store fragment to output (each thread writes 8 bytes)
    %tid = gpu.thread_id x
    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx2> %fragment, %out_ptr[%out_off_vgpr]
      : !vx2, !sx2[!v]
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

//   // Test @store_global_16x16xf32_C_fragment_wait: store C fragment to global
//   // Initialize accumulators with known values, then store using swizzled function
//   amdgcn.kernel @test_store_global_C_fragment_wait arguments <[
//     #amdgcn.buffer_arg<address_space = generic, access = read_write>
//   ]> attributes {shared_memory_size = 0 : i32} {
//     %out_ptr = amdgcn.load_arg 0 : !sx2
//     amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

//     %c0 = arith.constant 0 : index
//     %c16 = arith.constant 16 : index

//     // Initialize accumulator with lane_id (as float bits)
//     %lane = func.call @lane_id() : () -> index
//     %lane_i32 = arith.index_cast %lane : index to i32
//     %acc = func.call @init_vgprx4(%lane_i32) : (i32) -> !vx4

//     // Store using the library function
//     // i_pos=0, j_pos=0, N_SIZE=16, ii_pos=0, jj_pos=0
//     func.call @store_global_16x16xf32_C_fragment_wait(
//       %acc, %out_ptr, %c0, %c0, %c16, %c0, %c0
//     ) : (!vx4, !sx2, index, index, index, index, index) -> ()

//     amdgcn.end_kernel
//   }

//   // Test @global_load_dwordx2_wait + @ds_write_dwordx2_wait: decoupled global load and LDS write
//   // Load from global to memref, then write from memref to LDS, then read back from LDS
//   amdgcn.kernel @test_global_load_ds_write arguments <[
//     #amdgcn.buffer_arg<address_space = generic, access = read_only>,
//     #amdgcn.buffer_arg<address_space = generic, access = read_write>
//   ]> attributes {shared_memory_size = 512 : i32} {
//     %in_ptr = amdgcn.load_arg 0 : !sx2
//     %out_ptr = amdgcn.load_arg 1 : !sx2
//     amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %c2 = arith.constant 2 : index
//     %c4 = arith.constant 4 : index
//     %c16 = arith.constant 16 : index

//     // Allocate memref for intermediate storage and cast to dynamic
//     %memref_static = memref.alloca() : memref<1x1x!vx2>
//     %memref = memref.cast %memref_static : memref<1x1x!vx2> to memref<?x?x!vx2>

//     // Global load to memref
//     func.call @global_load_dwordx2_wait(
//       %in_ptr,           // ptr
//       %c0, %c0,          // i_pos, j_pos (major tile)
//       %c16,              // N_SIZE
//       %c0, %c0,          // ii_pos, jj_pos (minor tile)
//       %c1,               // NN (number of 16 tiles = 1)
//       %c0, %c0,          // memref indices
//       %memref            // memref
//     ) : (!sx2, index, index, index, index, index, index, index, index, memref<?x?x!vx2>) -> ()

//     // DS write from memref to LDS
//     func.call @ds_write_dwordx2_wait(
//       %c0,               // lds_base_off
//       %c0, %c0,          // ii_pos, jj_pos
//       %c16,              // NN_SIZE
//       %c1,               // NN (number of 16 tiles = 1)
//       %c0, %c0,          // memref indices
//       %memref            // memref
//     ) : (index, index, index, index, index, index, index, memref<?x?x!vx2>) -> ()

//     // Read back from LDS and store to output
//     %tid = gpu.thread_id x
//     %lane = func.call @lane_id() : () -> index
//     %iii, %jjj = func.call @lane_delinearize_2d(%c16, %c4) : (index, index) -> (index, index)
//     %jjj_pos = affine.apply affine_map<()[jjj] -> (jjj * 4)>()[%jjj]

//     %lds_off = affine.apply affine_map<()[iii, jjj_pos] -> ((iii * 16 + jjj_pos) * 2)>()[%iii, %jjj_pos]
//     %lds_off_i32 = arith.index_cast %lds_off : index to i32
//     %lds_off_vgpr = lsir.to_reg %lds_off_i32 : i32 -> !v

//     %dst = func.call @alloc_vgprx2() : () -> (!vx2)
//     %c0_i32 = arith.constant 0 : i32
//     %from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %dst, %lds_off_vgpr, offset = %c0_i32
//       : !v, i32 -> !vx2
//     amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

//     %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
//     %out_off_i32 = arith.index_cast %out_off : index to i32
//     %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
//     amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx2> %from_lds, %out_ptr[%out_off_vgpr]
//       : !vx2, !sx2[!v]
//     amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

//     amdgcn.end_kernel
//   }

}
