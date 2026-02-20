// Kittens GEMM kernel with scf.for K-loop: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
// Uses scf.for loop with iter_args for arbitrary K (must be divisible by 16)
//
// Loads at sched.stage=0, compute at sched.stage=4.
// The SCF pipeliner generates the prefetch prologue and drain epilogue
// automatically, giving 4 iterations of load-compute overlap.
//
// Template parameters:
//   {{K}}         - K dimension (e.g., 32, 64, 128)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_16x16xK target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16_future(!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T
  // Input: A (16xK f16, row-major), B (16xK f16, row-major)
  // Output: C (16x16 f32, row-major)
  amdgcn.kernel @gemm_16x16xK arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: loads at stage 0, compute at stage 4.
    // The pipeliner will peel 4 prologue iterations (loads only)
    // and 4 epilogue iterations (compute only).
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = arith.muli %k, %c16 : index

      // Stage 0: issue loads
      %A_fut = func.call @load_A_f16(%A_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = 0 : i32} : (!sx2, index, index, index) -> !future_global_read
      %B_fut = func.call @load_B_f16(%B_ptr, %c0, %k_offset, %stride_AB)
          {sched.stage = 0 : i32} : (!sx2, index, index, index) -> !future_global_read

      // Stage 4: wait + MFMA (consumes futures from 4 iterations ago)
      %new_acc = func.call @mfma_f32_16x16x16_f16_future(%A_fut, %B_fut, %acc)
          {sched.stage = 4 : i32} : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C) : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
