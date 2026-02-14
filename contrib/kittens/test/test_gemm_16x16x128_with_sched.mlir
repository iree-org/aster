// Kittens GEMM kernel: C = A @ B^T
// A: 16x128 (f16), B: 16x128 (f16), C: 16x16 (f32)
// Uses 8 MFMA iterations (K=16 each) to cover K=128.
//
// Structure: scf.for with sched.delay/rate + autoschedule + op-scheduling.
// Loads have delay=0, MFMA has delay=2, so loads run 2 iterations ahead
// of compute, keeping at most 4 loads in flight (2 A + 2 B).

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_16x16x128_sched target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16_future(!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  func.func private @gemm_loop(
    %A_ptr: !sx2,
    %B_ptr: !sx2,
    %C_ptr: !sx2
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    %stride_AB = arith.constant 256 : index  // 128 elements * 2 bytes per f16
    %stride_C  = arith.constant 64 : index   // 16 elements * 4 bytes per f32

    // Memrefs for SROA + mem2reg of futures and accumulators
    %A_futures = memref.alloca(%c8) : memref<?x!future_global_read>
    %B_futures = memref.alloca(%c8) : memref<?x!future_global_read>
    %C_accum  = memref.alloca() : memref<!rt_C_f32>

    // Initialize accumulator
    %C_init = func.call @zero_C() : () -> !rt_C_f32
    memref.store %C_init, %C_accum[] : memref<!rt_C_f32>

    scf.for %k = %c0 to %c8 step %c1 {
      // Column offset: k * 16
      %col = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Loads: delay=0, fires every iteration starting at k=0
      %A_fut = func.call @load_A_f16(%A_ptr, %c0, %col, %stride_AB)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (!sx2, index, index, index) -> !future_global_read
      memref.store %A_fut, %A_futures[%k] : memref<?x!future_global_read>

      %B_fut = func.call @load_B_f16(%B_ptr, %c0, %col, %stride_AB)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (!sx2, index, index, index) -> !future_global_read
      memref.store %B_fut, %B_futures[%k] : memref<?x!future_global_read>

      // MFMA: delay=2, fires every iteration starting at k=2
      // At k=2 it consumes the futures from k=0, etc.
      %A_ready = memref.load %A_futures[%k] : memref<?x!future_global_read>
      %B_ready = memref.load %B_futures[%k] : memref<?x!future_global_read>
      %C_prev = memref.load %C_accum[] : memref<!rt_C_f32>
      %C_next = func.call @mfma_f32_16x16x16_f16_future(%A_ready, %B_ready, %C_prev)
        {sched.delay = 2 : i64, sched.rate = 1 : i64}
        : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
      memref.store %C_next, %C_accum[] : memref<!rt_C_f32>

    } {sched.dims = array<i64: 8>}

    // Store result
    %C_final = memref.load %C_accum[] : memref<!rt_C_f32>
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C) : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token
    return
  }

  // GEMM kernel: C[16x16] = A[16x128] @ B[16x128]^T
  amdgcn.kernel @gemm_16x16x128_sched arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    func.call @gemm_loop(%A_ptr, %B_ptr, %C_ptr)
      : (!sx2, !sx2, !sx2) -> ()

    amdgcn.end_kernel
  }
}
