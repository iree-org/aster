// Test kernel for kittens @load_A_f16 and @store_A_f16 functions.
// Verifies round-trip: load from global -> store to global produces same data.

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!rt_A_f16 = !vx2

amdgcn.module @test_kittens_load_store_A target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !rt_A_f16
  func.func private @store_A_f16(!rt_A_f16, !sx2, index, index, index)

  // Test @load_A_f16 and @store_A_f16: load a 16x16 f16 tile and store it back
  // Input: 16x16 f16 matrix (256 elements = 512 bytes)
  // Output: 16x16 f16 matrix (256 elements = 512 bytes)
  amdgcn.kernel @test_load_store_A arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Position and stride for 16x16 tile at origin
    %c0 = arith.constant 0 : index
    %stride = arith.constant 32 : index  // 16 elements * 2 bytes per f16

    // Load tile using kittens function
    %tile = func.call @load_A_f16(%in_ptr, %c0, %c0, %stride) : (!sx2, index, index, index) -> !rt_A_f16

    // Store tile using kittens function
    func.call @store_A_f16(%tile, %out_ptr, %c0, %c0, %stride) : (!rt_A_f16, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
