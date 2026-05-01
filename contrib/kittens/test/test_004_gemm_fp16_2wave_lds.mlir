// 2-wave GEMM kernel with LDS (16x32 tiles) + AGPR accumulators:
// C[32x16] = A[32xK] @ B[16xK]^T
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
//
// 2x1 wave grid (waves_m=2, waves_n=1):
//   Wave 0: C[0:16,  0:16] = A[0:16,  :K] @ B[0:16, :K]^T
//   Wave 1: C[16:32, 0:16] = A[16:32, :K] @ B[0:16, :K]^T
//
// LDS layout: 3 tiles (A0, A1, B_shared), 1024 bytes each = 3072 bytes
//
// Template parameters:
//   {{K}}         - K dimension (must be divisible by 32)
//   {{K_TILES}}   - Number of K tiles = K / 32
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.module @kittens_gemm_2wave_lds target = #amdgcn.target<gfx942> {
  // From mlir_kernels/library/common/indexing.mlir
  func.func private @wave_id() -> index

  // From compute_16x16_f16.mlir (AGPR)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !aster_utils.any, index, index, index)
  func.func private @prepare_ptr(!sx2) -> !aster_utils.any

  // From lds_16x64_b.mlir
  func.func private @load_global_tile_16x64_b(!aster_utils.any, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @load_lds_A_swizzled(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled(index, index, index) -> !future_lds_read
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  // 2-wave GEMM kernel (128 threads = 2 waves) with LDS + AGPR
  amdgcn.kernel @gemm_2wave_lds arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_raw = amdgcn.load_arg 0 : !sx2
    %B_raw = amdgcn.load_arg 1 : !sx2
    %C_raw = amdgcn.load_arg 2 : !sx2
    amdgcn.s_waitcnt lgkmcnt = 0
    %A_ptr = func.call @prepare_ptr(%A_raw) : (!sx2) -> !aster_utils.any
    %B_ptr = func.call @prepare_ptr(%B_raw) : (!sx2) -> !aster_utils.any
    %C_ptr = func.call @prepare_ptr(%C_raw) : (!sx2) -> !aster_utils.any

    // Constants
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index  // bytes per f16 element
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index  // K1 byte offset within LDS row

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 32)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position: layout maps wave_id -> M tile offset
    %wid = func.call @wave_id() : () -> index
    %m_offset = layout.linearize %wid, #layout.strided_layout<[2] : [16]>

    // Allocate LDS: 3 tiles (A0 for wave 0, A1 for wave 1, B shared)
    %lds_a0_h = amdgcn.alloc_lds 1024
    %lds_A0 = amdgcn.get_lds_offset %lds_a0_h : index
    %lds_a1_h = amdgcn.alloc_lds 1024
    %lds_A1 = amdgcn.get_lds_offset %lds_a1_h : index
    %lds_b_h = amdgcn.alloc_lds 1024
    %lds_B = amdgcn.get_lds_offset %lds_b_h : index

    // Select this wave's A LDS buffer
    %lds_A_stride = affine.apply affine_map<()[a0, a1] -> (a1 - a0)>()[%lds_A0, %lds_A1]
    %lds_A = affine.apply affine_map<()[base, wid, stride] -> (base + wid * stride)>
        ()[%lds_A0, %wid, %lds_A_stride]

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Step 1: Global load 16x32 tiles (dwordx4) ===
      %A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %m_offset, %k_offset, %stride_AB)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %k_offset, %stride_AB)
          : (!aster_utils.any, index, index, index) -> !future_global_read

      // === Step 2: Store to LDS (2 tokens per tile) ===
      %tok_A0, %tok_A1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A, %A_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_B0, %tok_B1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B, %B_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)

      // === Step 3: Wait for LDS writes + cross-wave barrier ===
      amdgcn.wait deps %tok_A0 : !lds_write_token
      amdgcn.wait deps %tok_A1 : !lds_write_token
      amdgcn.wait deps %tok_B0 : !lds_write_token
      amdgcn.wait deps %tok_B1 : !lds_write_token
      amdgcn.s_barrier

      // === Step 4: K0 sub-tile (byte offset 0 within LDS row) ===
      %A0_future = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2) : (index, index, index) -> !future_lds_read
      %A0_tile = func.call @get_lds_read_value_vx2(%A0_future) : (!future_lds_read) -> !rt_A_f16
      %B0_future = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2) : (index, index, index) -> !future_lds_read
      %B0_tile = func.call @get_lds_read_value_vx2(%B0_future) : (!future_lds_read) -> !rt_B_f16
      %acc_k0 = func.call @mfma_f32_16x16x16_f16(%A0_tile, %B0_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // === Step 5: K1 sub-tile (byte offset 32 within LDS row) ===
      %A1_future = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2) : (index, index, index) -> !future_lds_read
      %A1_tile = func.call @get_lds_read_value_vx2(%A1_future) : (!future_lds_read) -> !rt_A_f16
      %B1_future = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2) : (index, index, index) -> !future_lds_read
      %B1_tile = func.call @get_lds_read_value_vx2(%B1_future) : (!future_lds_read) -> !rt_B_f16
      %acc_k1 = func.call @mfma_f32_16x16x16_f16(%A1_tile, %B1_tile, %acc_k0)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      scf.yield %acc_k1 : !rt_C_f32
    }

    // Fire-and-forget store
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C_final, %C_ptr, %m_offset, %c0, %stride_C)
        : (!rt_C_f32, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
