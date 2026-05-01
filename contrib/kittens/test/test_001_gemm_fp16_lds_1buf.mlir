// Kittens GEMM kernel with single-buffer LDS (16x32 tiles): C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
// Single buffer (no latency hiding) - memory bound performance.
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

amdgcn.module @kittens_gemm_16x16xK_lds_1buf target = #amdgcn.target<gfx942> {
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

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using single-buffer LDS + AGPR
  // LDS layout: 1024 bytes per tile (2x 512-byte XOR-swizzled 16x16 sub-tiles)
  amdgcn.kernel @gemm_16x16xK_lds_1buf arguments <[
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

    // Number of K tiles (K / 32, each tile covers 32 K-elements)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: one 16x32 tile for A, one for B (1024 bytes each)
    %lds_a_h = amdgcn.alloc_lds 1024
    %lds_A = amdgcn.get_lds_offset %lds_a_h : index
    %lds_b_h = amdgcn.alloc_lds 1024
    %lds_B = amdgcn.get_lds_offset %lds_b_h : index

    // Initialize accumulator to zero (AGPR)
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop: iterate over K dimension in 32-element tiles
    // Each iteration: load 16x32 tile, do 2 MFMAs (K0 and K1 sub-tiles)
    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      // k_offset = k * 32 (column offset in elements)
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Step 1: Global load 16x32 tiles (dwordx4) ===
      %A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %k_offset, %stride_AB)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %k_offset, %stride_AB)
          : (!aster_utils.any, index, index, index) -> !future_global_read

      // === Step 2: Store to LDS (2 tokens per tile: K0 and K1 halves) ===
      %tok_A0, %tok_A1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A, %A_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_B0, %tok_B1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B, %B_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)

      // === Step 3: Wait for all LDS writes ===
      amdgcn.wait deps %tok_A0 : !lds_write_token
      amdgcn.wait deps %tok_A1 : !lds_write_token
      amdgcn.wait deps %tok_B0 : !lds_write_token
      amdgcn.wait deps %tok_B1 : !lds_write_token

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

    // Fire-and-forget store (s_endpgm drains outstanding stores)
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !aster_utils.any, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
