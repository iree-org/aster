// 2-wave GEMM kernel with LDS: C[32x16] = A[32xK] @ B[16xK]^T
//
// 2x1 wave grid (waves_m=2, waves_n=1):
//   Wave 0: C[0:16,  0:16] = A[0:16,  :K] @ B[0:16, :K]^T
//   Wave 1: C[16:32, 0:16] = A[16:32, :K] @ B[0:16, :K]^T
//
// LDS layout: 3 tiles (A0, A1, B_shared), XOR swizzle (512 bytes/tile)
//   - Each wave loads its own A tile into its LDS buffer
//   - Both waves redundantly load the shared B tile (same data, same address)
//   - s_barrier synchronizes all 128 threads before LDS reads
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 16)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.module @kittens_gemm_2wave_lds target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From mlir_kernels/library/common/indexing.mlir
  func.func private @wave_id() -> index

  // From kittens/global_16x16_f16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir - XOR swizzle mode
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // 2-wave GEMM kernel (128 threads = 2 waves) with LDS
  // Input:  A [32xK f16, row-major], B [16xK f16, row-major]
  // Output: C [32x16 f32, row-major]
  //
  // LDS: 3 XOR-swizzle tiles x 512 bytes = 1,536 bytes total
  //   A0: wave 0's A tile
  //   A1: wave 1's A tile
  //   B:  shared B tile (both waves write same data)
  amdgcn.kernel @gemm_2wave_lds arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 1536 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position: m_offset = wave_id * 16
    %wid = func.call @wave_id() : () -> index
    %m_offset = affine.apply affine_map<()[wid] -> (wid * 16)>()[%wid]

    // Allocate LDS: 3 XOR-swizzle tiles (A0, A1, B_shared)
    %lds_A0_alloc = amdgcn.alloc_lds 512
    %lds_A1_alloc = amdgcn.alloc_lds 512
    %lds_B_alloc  = amdgcn.alloc_lds 512
    %lds_A0 = amdgcn.get_lds_offset %lds_A0_alloc : index
    %lds_A1 = amdgcn.get_lds_offset %lds_A1_alloc : index
    %lds_B  = amdgcn.get_lds_offset %lds_B_alloc  : index

    // Select this wave's A LDS buffer: wave 0 -> A0, wave 1 -> A1
    %lds_A_stride = affine.apply affine_map<()[a0, a1] -> (a1 - a0)>()[%lds_A0, %lds_A1]
    %lds_A = affine.apply affine_map<()[base, wid, stride] -> (base + wid * stride)>
        ()[%lds_A0, %wid, %lds_A_stride]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: each wave iterates over K tiles
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Step 1: Cooperative load Global -> LDS ===
      // Each wave loads its own A tile (wave 0 -> A0, wave 1 -> A1)
      %tok_A = func.call @load_global_to_lds_xor_swizzle_f16(%lds_A, %A_ptr, %m_offset, %k_offset, %stride_AB)
          : (index, !sx2, index, index, index) -> !lds_write_token

      // Both waves redundantly load shared B tile (same data, same LDS address)
      %tok_B = func.call @load_global_to_lds_xor_swizzle_f16(%lds_B, %B_ptr, %c0, %k_offset, %stride_AB)
          : (index, !sx2, index, index, index) -> !lds_write_token

      // === Step 2: Wait for LDS writes + cross-wave barrier ===
      amdgcn.wait deps %tok_A : !lds_write_token
      amdgcn.wait deps %tok_B : !lds_write_token
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier>

      // === Step 3: Load LDS -> Register ===
      %A_future = func.call @load_lds_A_xor_swizzle_f16(%lds_A)
          : (index) -> !future_lds_read
      %A_tile = func.call @get_lds_A_f16(%A_future)
          : (!future_lds_read) -> !rt_A_f16
      %B_future = func.call @load_lds_B_xor_swizzle_f16(%lds_B)
          : (index) -> !future_lds_read
      %B_tile = func.call @get_lds_B_f16(%B_future)
          : (!future_lds_read) -> !rt_B_f16

      // === Step 4: Compute ===
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %new_acc : !rt_C_f32
    }

    // Store result at this wave's row offset in C
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %m_offset, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
