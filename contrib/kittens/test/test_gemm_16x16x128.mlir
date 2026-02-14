// Kittens GEMM kernel: C = A @ B^T
// A: 16x128 (f16), B: 16x128 (f16), C: 16x16 (f32)
// Uses 8 MFMA iterations (K=16 each) to cover K=128.
//
// Structure: interleaved loads and compute, at most 4 loads in flight.
// Prefetch 2 iterations (4 loads), then steady-state: issue 2 loads, consume 2.
// Drain the last 2 iterations without issuing new loads.

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_16x16x128 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16_future(!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // GEMM kernel: C[16x16] = A[16x128] @ B[16x128]^T
  amdgcn.kernel @gemm_16x16x128 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0   = arith.constant 0 : index
    %c16  = arith.constant 16 : index
    %c32  = arith.constant 32 : index
    %c48  = arith.constant 48 : index
    %c64  = arith.constant 64 : index
    %c80  = arith.constant 80 : index
    %c96  = arith.constant 96 : index
    %c112 = arith.constant 112 : index

    // Strides in bytes
    %stride_AB = arith.constant 256 : index  // 128 elements * 2 bytes per f16
    %stride_C  = arith.constant 64 : index   // 16 elements * 4 bytes per f32

    %C0 = func.call @zero_C() : () -> !rt_C_f32

    // ====================================================================
    // Prefetch: issue 2 iterations of loads (4 loads in flight)
    // ====================================================================
    %A0 = func.call @load_A_f16(%A_ptr, %c0, %c0,  %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B0 = func.call @load_B_f16(%B_ptr, %c0, %c0,  %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %A1 = func.call @load_A_f16(%A_ptr, %c0, %c16, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B1 = func.call @load_B_f16(%B_ptr, %c0, %c16, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // ====================================================================
    // Steady state: consume oldest pair, issue next pair (4 in flight)
    // ====================================================================

    // k=0: consume (A0,B0), issue (A2,B2)
    %C1 = func.call @mfma_f32_16x16x16_f16_future(%A0, %B0, %C0) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A2 = func.call @load_A_f16(%A_ptr, %c0, %c32, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B2 = func.call @load_B_f16(%B_ptr, %c0, %c32, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // k=1: consume (A1,B1), issue (A3,B3)
    %C2 = func.call @mfma_f32_16x16x16_f16_future(%A1, %B1, %C1) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A3 = func.call @load_A_f16(%A_ptr, %c0, %c48, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B3 = func.call @load_B_f16(%B_ptr, %c0, %c48, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // k=2: consume (A2,B2), issue (A4,B4)
    %C3 = func.call @mfma_f32_16x16x16_f16_future(%A2, %B2, %C2) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A4 = func.call @load_A_f16(%A_ptr, %c0, %c64, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B4 = func.call @load_B_f16(%B_ptr, %c0, %c64, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // k=3: consume (A3,B3), issue (A5,B5)
    %C4 = func.call @mfma_f32_16x16x16_f16_future(%A3, %B3, %C3) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A5 = func.call @load_A_f16(%A_ptr, %c0, %c80, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B5 = func.call @load_B_f16(%B_ptr, %c0, %c80, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // k=4: consume (A4,B4), issue (A6,B6)
    %C5 = func.call @mfma_f32_16x16x16_f16_future(%A4, %B4, %C4) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A6 = func.call @load_A_f16(%A_ptr, %c0, %c96,  %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B6 = func.call @load_B_f16(%B_ptr, %c0, %c96,  %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // k=5: consume (A5,B5), issue (A7,B7) -- last loads
    %C6 = func.call @mfma_f32_16x16x16_f16_future(%A5, %B5, %C5) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
    %A7 = func.call @load_A_f16(%A_ptr, %c0, %c112, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B7 = func.call @load_B_f16(%B_ptr, %c0, %c112, %stride_AB) : (!sx2, index, index, index) -> !future_global_read

    // ====================================================================
    // Drain: consume remaining pairs, no new loads
    // ====================================================================

    // k=6: consume (A6,B6)
    %C7 = func.call @mfma_f32_16x16x16_f16_future(%A6, %B6, %C6) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

    // k=7: consume (A7,B7)
    %C8 = func.call @mfma_f32_16x16x16_f16_future(%A7, %B7, %C7) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

    // ====================================================================
    // Store result
    // ====================================================================
    %store_tok = func.call @store_C_f32(%C8, %C_ptr, %c0, %c0, %stride_C) : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
