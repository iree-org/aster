// RUN: aster-opt %s \
// RUN:   --aster-selective-inlining \
// RUN:   --aster-op-scheduling \
// RUN:   --aster-selective-inlining="allow-scheduled-calls=true" \
// RUN:   --cse --canonicalize --sroa \
// RUN:   --cse --canonicalize --amdgcn-mem2reg \
// RUN:   --aster-selective-inlining="allow-scheduled-calls=true" \
// RUN:   --cse --canonicalize --symbol-dce \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-optimize-straight-line-waits \
// RUN:   --aster-to-int-arith \
// RUN:   --aster-optimize-arith \
// RUN:   --aster-amdgcn-set-abi \
// RUN:   --aster-codegen \
// RUN:   --canonicalize \
// RUN:   --canonicalize \
// RUN:   --aster-to-amdgcn \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops,amdgcn-register-allocation)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-nop-insertion \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Note: Computation shape is C[1x1] <- A[1x1] * B[1x1] + C[1x1]
// CHECK-LABEL: test_matmul_kernel:
// CHECK:   s_load_dwordx2 [[A_ptr:.*]], s[0:1], 0
// CHECK:   s_load_dwordx2 [[B_ptr:.*]], s[0:1], 8
// CHECK:   s_load_dwordx2 [[C_ptr:.*]], s[0:1], 16
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// CHECK:   v_lshlrev_b32_e64 [[tidx_times_8:.*]], 3, v0
// CHECK:   global_load_dwordx2 [[A:v\[.*\]]], [[tidx_times_8]], [[A_ptr]]
// CHECK:   global_load_dwordx2 [[B:v\[.*\]]], [[tidx_times_8]], [[B_ptr]]
// CHECK:   s_waitcnt vmcnt(1) expcnt(0) lgkmcnt(0)
// CHECK:   ds_write_b64 [[tidx_times_8]], [[A]]
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(1)
// CHECK:   ds_write_b64 [[tidx_times_8]], [[B]] offset: 512
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(1)
// CHECK:   ds_read_b64 [[A]], [[tidx_times_8]]
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(1)
// CHECK:   ds_read_b64 [[B]], [[tidx_times_8]] offset: 512
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// CHECK:   v_mfma_f32_16x16x16_f16 [[C:v\[.*\]]], [[A]], [[B]], [[C]]
// CHECK:   v_lshlrev_b32_e64 [[tidx_times_16:.*]], 4, v0
// CHECK:   global_store_dwordx4 [[tidx_times_16]], [[C]], [[C_ptr]]
// CHECK:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// CHECK:   s_endpgm

amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Unified global load function
  func.func private @global_load_body(
    %threadidx_x: index,
    %sz0: index,
    %sz1: index,
    %st1: index,
    %global: !amdgcn.sgpr_range<[? + 2]>,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
  ) {
    // Allocate registers for matrix tile
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr1 = amdgcn.alloca : !amdgcn.vgpr
    %range = amdgcn.make_register_range %vgpr0, %vgpr1 : !amdgcn.vgpr, !amdgcn.vgpr

    // Calculate offset (sz0 * st1 + sz1) * 8 bytes for dwordx2
    %offset_index = affine.apply affine_map<(d0, d1, d2)[s0] -> ((d0 * s0 + d1 + d2) * 8)>(%sz0, %sz1, %threadidx_x)[%st1]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // Global load
    %c0_load = arith.constant 0 : i32
    %loaded, %tok = amdgcn.load global_load_dwordx2 dest %range addr %global offset d(%offset_vgpr) + c(%c0_load) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.read_token<flat>

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Store loaded value to memref for use in next block
    memref.store %loaded, %memref[%sz0, %sz1] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    return
  }

  // Unified global store function
  func.func private @global_store_body(
    %threadidx_x: index,
    %sz0: index,
    %sz1: index,
    %st1: index,
    %c_memref: memref<?x?x!amdgcn.vgpr_range<[? + 4]>>,
    %c_global: !amdgcn.sgpr_range<[? + 2]>
  ) {
    // Load from c_memref for SROA + MEM2REG
    %c_value = memref.load %c_memref[%sz0, %sz1]
      : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>

    // Calculate offset (sz0 * st1 + sz1) * 16 bytes for dwordx4
    %offset_index = affine.apply affine_map<(d0, d1, d2)[s0] -> ((d0 * s0 + d1 + d2) * 16)>(%sz0, %sz1, %threadidx_x)[%st1]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // Store vGPR range directly to global memory
    %c0_store = arith.constant 0 : i32
    %tok = amdgcn.store global_store_dwordx4 data %c_value addr %c_global offset d(%offset_vgpr) + c(%c0_store) : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<flat>
    return
  }

  // Unified DS read function
  func.func private @ds_read_body(
    %threadidx_x: index,
    %sz0: index,
    %sz1: index,
    %st1: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    // Allocate registers for matrix tile
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr
    %vgpr1 = amdgcn.alloca : !amdgcn.vgpr
    %range = amdgcn.make_register_range %vgpr0, %vgpr1 : !amdgcn.vgpr, !amdgcn.vgpr

    // Calculate offset (sz0 * st1 + sz1) * 8 bytes for dwordx2
    %offset_index = affine.apply affine_map<(d0, d1, d2)[s0] -> ((d0 * s0 + d1 + d2) * 8)>(%sz0, %sz1, %threadidx_x)[%st1]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // DS read from LDS
    %from_lds, %tok = amdgcn.load ds_read_b64 dest %range addr %offset_vgpr offset c(%lds_offset) : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>

    // Wait for LDS read
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Store to memref for later use
    memref.store %from_lds, %memref[%sz0, %sz1] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    return
  }

  // Unified DS write function
  func.func private @ds_write_body(
    %threadidx_x: index,
    %sz0: index,
    %sz1: index,
    %st1: index,
    %memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %lds_offset: i32
  ) {
    // Load the value from memref (stored by part 1)
    %loaded = memref.load %memref[%sz0, %sz1] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>

    // Calculate offset (sz0 * st1 + sz1) * 8 bytes for dwordx2

    %offset_index = affine.apply affine_map<(d0, d1, d2)[s0] -> ((d0 * s0 + d1 + d2) * 8)>(%sz0, %sz1, %threadidx_x)[%st1]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !amdgcn.vgpr

    // DS write to LDS
    %tok = amdgcn.store ds_write_b64 data %loaded addr %offset_vgpr offset c(%lds_offset) : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>

    // Wait for LDS write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
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

  // Initialize the C matrix to zero
  func.func private @zero_init_body(
    %sz0: index,
    %sz1: index,
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
    memref.store %c0_vgpr_range, %memref[%sz0, %sz1] : memref<?x?x!amdgcn.vgpr_range<[? + 4]>>
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
    %c512_i32 = arith.constant 512 : i32

    // Calculate MNK iterations using affine.apply with symbols
    %MNK = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%M, %N, %K]

    %threadidx_x = gpu.thread_id x

    // Loop over linearized index
    scf.for %i = %c0 to %MNK step %c1 {
      // Delinearize index i into (m, n, k) using basis (M, N, K)
      %m, %n, %k = affine.delinearize_index %i into (%M, %N, %K) : index, index, index

      // Part 1a: Global load for A
      func.call @global_load_body(%threadidx_x, %m, %k, %K, %a_global, %a_memref)
        : (index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>) -> ()

      // Part 2a: Global load for B
      func.call @global_load_body(%threadidx_x, %k, %n, %N, %b_global, %b_memref)
        : (index, index, index, index, !amdgcn.sgpr_range<[? + 2]>, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>) -> ()

      // Part 1b: DS write for A
      func.call @ds_write_body(%threadidx_x, %m, %k, %K, %a_memref, %c0_i32)
        : (index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()

      // Part 2b: DS write for B
      func.call @ds_write_body(%threadidx_x, %k, %n, %N, %b_memref, %c512_i32)
        : (index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()

      // Part 1c: DS read for A
      func.call @ds_read_body(%threadidx_x, %m, %k, %K, %a_memref, %c0_i32)
        : (index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()

      // Part 2c: DS read for B
      func.call @ds_read_body(%threadidx_x, %k, %n, %N, %b_memref, %c512_i32)
        : (index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 2]>>, i32) -> ()

      // Initialize the C matrix to zero
      func.call @zero_init_body(%m, %n, %c_memref)
        : (index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>) -> ()

      // Call simple_mfma with delinearized indices
      func.call @simple_mfma(%m, %n, %k, %a_memref, %b_memref, %c_memref)
        : (index, index, index,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.vgpr_range<[? + 4]>>) -> ()

      // Store C to global memory
      func.call @global_store_body(%threadidx_x, %m, %n, %N, %c_memref, %c_global)
        : (index, index, index, index, memref<?x?x!amdgcn.vgpr_range<[? + 4]>>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    } {sched.dims = array<i64: 1, 1, 1>}

    return
  }

  // Test function that calls matmul_loop with specific dimensions
  func.func private @test_matmul(%a_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %b_global: !amdgcn.sgpr_range<[? + 2]>,
                                  %c_global: !amdgcn.sgpr_range<[? + 2]>) {
    // Set dimensions: M=1, N=1, K=1
    %c1 = arith.constant 1 : index

    // Call matmul_loop with these dimensions
    func.call @matmul_loop(%c1, %c1, %c1, %a_global, %b_global, %c_global)
      : (index, index, index, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()

    return
  }

  amdgcn.kernel @test_matmul_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 1024 : i32} {
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
