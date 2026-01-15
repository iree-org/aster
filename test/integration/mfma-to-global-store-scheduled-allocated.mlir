// RUN: aster-opt %s --inline \
// RUN:   --amdgcn-instruction-scheduling-autoschedule --aster-op-scheduling \
// RUN:   --cse --canonicalize --sroa \
// RUN:   --cse --canonicalize --amdgcn-mem2reg \
// RUN:   --cse --canonicalize --symbol-dce \
// RUN:   --amdgcn-register-allocation \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// Note: Computation shape is C[4x1] <- A[4x3] * B[3x1] + C[4x1]
// CHECK-LABEL: test_matmul_kernel:
//       CHECK:   v_mfma_f32_16x16x16_f16 [[AC0:a.*]], [[VA00:v.*]], [[VB0:v.*]], [[AC0]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC0]], [[VA01:v.*]], [[VB1:v.*]], [[AC0]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC0]], [[VA02:v.*]], [[VB2:v.*]], [[AC0]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC1:a.*]], [[VA10:v.*]], [[VB0]], [[AC1]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC1]], [[VA11:v.*]], [[VB1]], [[AC1]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC1]], [[VA12:v.*]], [[VB2]], [[AC1]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC2:a.*]], [[VA20:v.*]], [[VB0]], [[AC2]]
//  CHECK-NEXT:   global_store_dwordx4 {{.*}}, [[AC0]], off
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC2]], [[VA21:v.*]], [[VB1]], [[AC2]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC2]], [[VA22:v.*]], [[VB2]], [[AC2]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC3:a.*]], [[VA30:v.*]], [[VB0]], [[AC3]]
//  CHECK-NEXT:   global_store_dwordx4 {{.*}}, [[AC1]], off
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC3]], [[VA31:v.*]], [[VB1]], [[AC3]]
//  CHECK-NEXT:   v_mfma_f32_16x16x16_f16 [[AC3]], [[VA32:v.*]], [[VB2]], [[AC3]]
//  CHECK-NEXT:   global_store_dwordx4 {{.*}}, [[AC2]], off
//  CHECK-NEXT:   global_store_dwordx4 {{.*}}, [[AC3]], off
//  CHECK-NEXT:   s_endpgm

amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Function taking indices and memrefs of register ranges
  func.func private @simple_mfma(
    %m: index,
    %n: index,
    %k: index,
    %a_memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %b_memref: memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
    %c_memref: memref<?x?x!amdgcn.agpr_range<[? + 4]>>
  ) {
    // Load register ranges from memrefs at specified indices for SROA + MEM2REG
    %a = memref.load %a_memref[%m, %k] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %b = memref.load %b_memref[%k, %n] : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %c = memref.load %c_memref[%m, %n] : memref<?x?x!amdgcn.agpr_range<[? + 4]>>

    // Perform MFMA operation: C = A * B + C
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16>
      %c, %a, %b, %c : <[? + 2]>, <[? + 2]>, !amdgcn.agpr_range<[? + 4]>
      -> !amdgcn.agpr_range<[? + 4]>

    // Only schedule this operation and let autoschedule handle the rest.
    // Store result back to memref at [m, n] for SROA + MEM2REG
    memref.store %result, %c_memref[%m, %n]
        {sched.delay = 1 : i64, sched.rate = 2 : i64, sched.permutation = array<i64: 0, 2, 1>}
      : memref<?x?x!amdgcn.agpr_range<[? + 4]>>

    return
  }

  // Main function that allocates memrefs and loops over MxNxK
  func.func private @matmul_loop(%M: index, %N: index, %K: index, %c_global: !amdgcn.vgpr_range<[10 : 12]>) {
    // Allocate memrefs for A, B, and C matrices
    %a_memref = memref.alloca(%M, %K) : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %b_memref = memref.alloca(%K, %N) : memref<?x?x!amdgcn.vgpr_range<[? + 2]>>
    %c_memref = memref.alloca(%M, %N) : memref<?x?x!amdgcn.agpr_range<[? + 4]>>

    // Constants for loop
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Calculate total iterations using affine.apply with symbols
    %total = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 * s2)>()[%M, %N, %K]

    // Loop over linearized index
    scf.for %i = %c0 to %total step %c1 {
      // Delinearize index i into (m, n, k) using basis (M, N, K)
      %m, %n, %k = affine.delinearize_index %i into (%M, %N, %K) : index, index, index

      // Call simple_mfma with delinearized indices
      func.call @simple_mfma(%m, %n, %k, %a_memref, %b_memref, %c_memref)
        : (index, index, index,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.vgpr_range<[? + 2]>>,
           memref<?x?x!amdgcn.agpr_range<[? + 4]>>) -> ()

      // If last K iteration, load from c_memref and store to global memory
      %k_minus_1 = arith.subi %K, %c1  : index
      %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
      // Only schedule this operation and let autoschedule handle the rest.
      scf.if %is_last_k {
        // Load from c_memref for SROA + MEM2REG
        %c_value = memref.load %c_memref[%m, %n]
        : memref<?x?x!amdgcn.agpr_range<[? + 4]>>
        // Store AGPR range directly to global memory
        %c0_gs = arith.constant 0 : i32
        %tok = amdgcn.store global_store_dwordx4 data %c_value addr %c_global offset c(%c0_gs) : ins(!amdgcn.agpr_range<[? + 4]>, !amdgcn.vgpr_range<[10 : 12]>, i32) -> !amdgcn.write_token<flat>
      } {sched.delay = 10 : i64, sched.rate = 2 : i64, sched.permutation = array<i64: 0, 2, 1>}

    } {sched.dims = array<i64: 4, 1, 3>}

    return
  }

  // Test function that calls matmul_loop with specific dimensions
  func.func private @test_matmul() {
    %c_memref = memref.alloca() : memref<!amdgcn.vgpr_range<[10 : 12]>>
    %c_global = memref.load %c_memref[] : memref<!amdgcn.vgpr_range<[10 : 12]>>

    // Set dimensions: M=4, N=1, K=3
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Call matmul_loop with these dimensions
    func.call @matmul_loop(%c4, %c1, %c3, %c_global) : (index, index, index, !amdgcn.vgpr_range<[10 : 12]>) -> ()

    return
  }

  amdgcn.kernel @test_matmul_kernel {
    func.call @test_matmul() : () -> ()
    amdgcn.end_kernel
  }
}
