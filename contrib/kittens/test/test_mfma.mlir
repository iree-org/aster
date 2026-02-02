// Test kernel for kittens @mfma_f32_16x16x16_f16 and @store_C_f32 functions.
// Verifies MFMA correctness: D = A @ B^T + C

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4

amdgcn.module @test_kittens_mfma target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !rt_A_f16
  func.func private @load_B_f16(!sx2, index, index, index) -> !rt_B_f16
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index)

  // Test @mfma_f32_16x16x16_f16: compute D = A @ B^T + 0
  // Input: A (16x16 f16), B (16x16 f16)
  // Output: D (16x16 f32)
  amdgcn.kernel @test_mfma arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %D_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Position and stride for 16x16 tiles at origin
    %c0 = arith.constant 0 : index
    %stride_f16 = arith.constant 32 : index  // 16 elements * 2 bytes per f16
    %stride_f32 = arith.constant 64 : index  // 16 elements * 4 bytes per f32

    // Load A and B tiles
    %A = func.call @load_A_f16(%A_ptr, %c0, %c0, %stride_f16) : (!sx2, index, index, index) -> !rt_A_f16
    %B = func.call @load_B_f16(%B_ptr, %c0, %c0, %stride_f16) : (!sx2, index, index, index) -> !rt_B_f16

    // Initialize C to zero
    %C = func.call @zero_C() : () -> !rt_C_f32

    // Perform MFMA: D = A @ B^T + C
    %D = func.call @mfma_f32_16x16x16_f16(%A, %B, %C) : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

    // Store D using kittens store function
    func.call @store_C_f32(%D, %D_ptr, %c0, %c0, %stride_f32) : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
