// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/4/g' -e 's/{{SIZE_N}}/4/g' -e 's/{{SIZE_K}}/4/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/../library/common/indexing.mlir" \
// RUN: | FileCheck %s

// Minimal 1-D copy kernel using dwordx4 (16 bytes per thread)

// CHECK-LABEL: amdgcn.module
amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From indexing.mlir
  func.func private @index_bxmxnxk_16x16x16_f16f16f32(index, index, index, index, index, index, index, index, index) -> index

  //
  // Unified global load function
  //
  func.func private @global_load_body(
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %global: !amdgcn.sgpr_range<[? + 2]>,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
  ) {
    // Allocate registers for matrix tile
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr1 = amdgcn.alloca : !amdgcn.vgpr
    %range = amdgcn.make_register_range %vgpr0, %vgpr1 : !amdgcn.vgpr, !amdgcn.vgpr

    // Compute byte offset using library function
    // tile_size = 16*16*2 = 512 (f16 input), lane_stride = 8 (dwordx2)
    %c512 = arith.constant 512 : index
    %c8 = arith.constant 8 : index
    %offset = func.call @index_bxmxnxk_16x16x16_f16f16f32(
      %bidx, %tidx, %i, %j, %szI, %szJ, %bdimx, %c512, %c8
    ) : (index, index, index, index, index, index, index, index, index) -> index
    %offset_vgpr = lsir.to_reg %offset : index -> !amdgcn.vgpr

    // Global load
    %c0_load = arith.constant 0 : i32
    %loaded, %tok = amdgcn.load global_load_dwordx2 dest %range addr %global offset d(%offset_vgpr) + c(%c0_load) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Store loaded value to memref for use in next block
    memref.store %loaded, %memref[%i, %j] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    return
  }

  func.func private @global_load_body_if(
    %cond: i1,
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %global: !amdgcn.sgpr_range<[? + 2]>,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
  ) {
    scf.if %cond {
      func.call @global_load_body(%tidx, %bidx, %bdimx, %gdimx, %i, %j, %szI, %szJ, %global, %memref)
        : (index, index, index, index, index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>) -> ()
    }
    return
  }

  //
  // Zero init function
  //
  func.func private @zero_init_body(
    %i: index,
    %j: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>
  ) {
    // Implicit i32 0 <=> f32 0.0
    %c0 = arith.constant 0 : i32
    %alloc = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_1 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_2 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr_3 = amdgcn.alloca : !amdgcn.vgpr
    %c0_vgpr_0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_0, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c0_vgpr_1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_1, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c0_vgpr_2 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_2, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c0_vgpr_3 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_3, %c0 : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
    %c0_vgpr_range = amdgcn.make_register_range %c0_vgpr_0, %c0_vgpr_1, %c0_vgpr_2, %c0_vgpr_3
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    memref.store %c0_vgpr_range, %memref[%i, %j] : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>
    return
  }

  func.func private @zero_init_body_if(
    %cond: i1,
    %i: index,
    %j: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>
  ) {
    scf.if %cond {
      func.call @zero_init_body(%i, %j, %memref)
        : (index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>) -> ()
    }
    return
  }

  //
  // Unified global store function
  //
  func.func private @global_store_body(
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %c_memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>,
    %c_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    // Load from c_memref for SROA + MEM2REG
    %c_value = memref.load %c_memref[%i, %j]
      : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>

    // Compute byte offset using library function
    // tile_size = 16*16*4 = 1024 (f32 output), lane_stride = 16 (dwordx4)
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %offset = func.call @index_bxmxnxk_16x16x16_f16f16f32(
      %bidx, %tidx, %i, %j, %szI, %szJ, %bdimx, %c1024, %c16
    ) : (index, index, index, index, index, index, index, index, index) -> index
    %offset_vgpr = lsir.to_reg %offset : index -> !amdgcn.vgpr

    // Store vGPR range directly to global memory
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %c_value addr %c_global offset d(%offset_vgpr) + c(%c0_store) : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<flat>
    return
  }

  func.func private @global_store_body_if(
    %cond: i1,
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %c_memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>,
    %c_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    scf.if %cond {
      func.call @global_store_body(%tidx, %bidx, %bdimx, %gdimx, %i, %j, %szI, %szJ, %c_memref, %c_global)
        : (index, index, index, index, index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>, !amdgcn.sgpr_range<[? + 2]>) -> ()
    }
    return
  }

  // Unified DS read function
  func.func private @ds_read_body(
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    // Allocate registers for matrix tile
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr1 = amdgcn.alloca : !amdgcn.vgpr
    %range = amdgcn.make_register_range %vgpr0, %vgpr1 : !amdgcn.vgpr, !amdgcn.vgpr

    // Workgroup level (irrelevant for this kernel) :
    // Wave level:
    //   read from [waveidx_x] with stride [ szI * szJ * 16 * 16 * 2] bytes.
    // Thread level:
    //   read from [i, j] with stride [ szJ * 16 * 16 * 2, 16 * 16 * 2] bytes.
    //   read from [lidx] with stride [ (16 * 16 * 2) / 64 = 8] bytes.
    %widx = affine.apply affine_map<()[tidx] -> (tidx floordiv 64)>()[%tidx]
    %lidx = affine.apply affine_map<()[tidx] -> (tidx mod 64)>()[%tidx]
    %offset = affine.apply affine_map<
      (widx, i, j, lidx)[szI, szJ]
        -> (widx * szI * szJ * 16 * 16 * 2 +
                     i * szJ * 16 * 16 * 2 +
                           j * 16 * 16 * 2 +
                                  lidx * 8)>
      (%widx, %i, %j, %lidx)[%szI, %szJ]
    %offset_vgpr = lsir.to_reg %offset : index -> !amdgcn.vgpr

    // DS read from LDS
    %from_lds, %tok = amdgcn.load ds_read_b64 dest %range addr %offset_vgpr offset c(%lds_offset) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // Wait for LDS read
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Store to memref for later use
    memref.store %from_lds, %memref[%i, %j] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    return
  }

  func.func private @ds_read_body_if(
    %cond: i1,
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    scf.if %cond {
      func.call @ds_read_body(%tidx, %bidx, %bdimx, %gdimx, %i, %j, %szI, %szJ, %memref, %lds_offset)
        : (index, index, index, index, index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()
    }
    return
  }

  // Unified DS write function
  func.func private @ds_write_body(
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    // Load the value from memref (stored by part 1)
    %loaded = memref.load %memref[%i, %j] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>

    // Workgroup level (irrelevant for this kernel) :
    // Wave level:
    //   write to [waveidx_x] with stride [ szI * szJ * 16 * 16 * 2] bytes.
    // Thread level:
    //   write to [i, j] with stride [ szJ * 16 * 16 * 2, 16 * 16 * 2] bytes.
    //   write to [lidx] with stride [ (16 * 16 * 2) / 64 = 8] bytes.
    %widx = affine.apply affine_map<()[tidx] -> (tidx floordiv 64)>()[%tidx]
    %lidx = affine.apply affine_map<()[tidx] -> (tidx mod 64)>()[%tidx]
    %offset = affine.apply affine_map<
      (widx, i, j, lidx)[szI, szJ]
        -> (widx * szI * szJ * 16 * 16 * 2 +
                     i * szJ * 16 * 16 * 2 +
                           j * 16 * 16 * 2 +
                                  lidx * 8)>
      (%widx, %i, %j, %lidx)[%szI, %szJ]
    %offset_vgpr = lsir.to_reg %offset : index -> !amdgcn.vgpr

    // DS write to LDS
    %tok = amdgcn.store ds_write_b64 data %loaded addr %offset_vgpr offset c(%lds_offset) : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Wait for LDS write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return
  }

  func.func private @ds_write_body_if(
    %cond: i1,
    %tidx: index,
    %bidx: index,
    %bdimx: index,
    %gdimx: index,
    %i: index,
    %j: index,
    %szI: index,
    %szJ: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    scf.if %cond {
      func.call @ds_write_body(%tidx, %bidx, %bdimx, %gdimx, %i, %j, %szI, %szJ, %memref, %lds_offset)
        : (index, index, index, index, index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()
    }
    return
  }

  // Function taking indices and memrefs of register ranges
  func.func private @simple_mfma(
    %m: index,
    %n: index,
    %k: index,
    %a_memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %b_memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %c_memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>
  ) {
    // Load register ranges from memrefs at specified indices for SROA + MEM2REG
    %a = memref.load %a_memref[%m, %k] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %b = memref.load %b_memref[%k, %n] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %c = memref.load %c_memref[%m, %n] : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>

    // Perform MFMA operation: C = A * B + C
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
      %c, %a, %b, %c : <[? + 2]>, <[? + 2]>, !amdgcn.vgpr_range<[? + 4]>
      -> !amdgcn.vgpr_range<[? + 4]>

    // Store result back to memref at [m, n] for SROA + MEM2REG
    memref.store %result, %c_memref[%m, %n]
      : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>

    return
  }

  // Main function that allocates memrefs and loops over MxNxK
  func.func private @matmul_loop(%M: index, %N: index, %K: index,
                                  %a_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %b_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %c_global: !amdgcn.sgpr_range<[? + 2]>) {
    // Allocate memrefs for A, B, and C matrices
    %a_memref = memref.alloca(%M, %K) : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %b_memref = memref.alloca(%K, %N) : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %c_memref = memref.alloca(%M, %N) : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>

    // Constants for loop
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %clds_b_shift_i32 = arith.constant {{LDS_B_SHIFT}} : i32

    // Calculate MNK iterations using affine.apply with symbols
    %MNK = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%M, %N, %K]

    // Get thread/block indices
    %tidx = gpu.thread_id x
    %bidx = gpu.block_id x
    %bdimx = gpu.block_dim x
    %gdimx = gpu.grid_dim x

    // Loop over linearized index
    scf.for %i = %c0 to %MNK step %c1 {
      // Delinearize index i into (m, n, k) using basis (M, N, K)
      %m, %n, %k = affine.delinearize_index %i into (%M, %N, %K) : index, index, index

      // Part 1a: Global load for A
      // If first N iteration, load A from global memory
      %is_first_n = arith.cmpi eq, %n, %c0 : index
      func.call @global_load_body_if(%is_first_n, %tidx, %bidx, %bdimx, %gdimx, %m, %k, %M, %K, %a_global, %a_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64, sched.permutation = array<i64: 0, 2, 1>}
        : (i1, index, index, index, index, index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>) -> ()
      // Part 2a: Global load for B
      // If first M iteration, load B from global memory
      %is_first_m = arith.cmpi eq, %m, %c0 : index
      func.call @global_load_body_if(%is_first_m, %tidx, %bidx, %bdimx, %gdimx, %k, %n, %K, %N, %b_global, %b_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64, sched.permutation = array<i64: 0, 2, 1>}
        : (i1, index, index, index, index, index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>) -> ()
      // Part 3a: Zero init for C
      %is_first_k = arith.cmpi eq, %k, %c0 : index
      func.call @zero_init_body_if(%is_first_k, %m, %n, %c_memref)
          {sched.delay = 0 : i64, sched.rate = 1 : i64, sched.permutation = array<i64: 0, 2, 1>}
        : (i1, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>) -> ()

      // Call simple_mfma with delinearized indices
      func.call @simple_mfma(%m, %n, %k, %a_memref, %b_memref, %c_memref)
        {sched.delay = 5 : i64, sched.rate = 1 : i64, sched.permutation = array<i64: 0, 2, 1>}
        : (index, index, index,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.vgpr_range<[? + 4]>>) -> ()

      // If last K iteration, store C to global memory
      %k_minus_1 = arith.subi %K, %c1 : index
      %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
      func.call @global_store_body_if(%is_last_k, %tidx, %bidx, %bdimx, %gdimx, %m, %n, %M, %N, %c_memref, %c_global)
        {sched.delay = 20 : i64, sched.rate = 1 : i64, sched.permutation = array<i64: 0, 2, 1>}
        : (i1, index, index, index, index, index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    } {sched.dims = array<i64: {{SIZE_M}}, {{SIZE_N}}, {{SIZE_K}}>}

    return
  }

  // Test function that calls matmul_loop with specific dimensions
  func.func private @test_matmul(%a_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %b_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %c_global: !amdgcn.sgpr_range<[? + 2]>) {
    %m = arith.constant {{SIZE_M}} : index
    %n = arith.constant {{SIZE_N}} : index
    %k = arith.constant {{SIZE_K}} : index

    // Call matmul_loop with these dimensions
    func.call @matmul_loop(%m, %n, %k, %a_global, %b_global, %c_global)
      : (index, index, index, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    return
  }

  amdgcn.kernel @test_matmul_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = {{LDS_SIZE}} : i32} {
    // <START Kernel ABI>
    %a_ptr_s = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %b_ptr_s = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %c_ptr_s = amdgcn.load_arg 2 : !amdgcn.sgpr_range<[? + 2]>
    %a_ptr, %b_ptr, %c_ptr = lsir.assume_noalias %a_ptr_s, %b_ptr_s, %c_ptr_s
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>)

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    // <END Kernel ABI>

    func.call @test_matmul(%a_ptr, %b_ptr, %c_ptr)
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    amdgcn.end_kernel
  }
}
