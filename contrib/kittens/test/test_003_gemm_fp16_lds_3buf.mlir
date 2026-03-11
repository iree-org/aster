// Kittens GEMM kernel with triple-buffer LDS (16x32 tiles) + AGPR: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
// Triple buffering adds a third buffer beyond double-buffering.
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

amdgcn.module @kittens_gemm_16x16xK_lds_3buf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From compute_16x16_f16.mlir (AGPR)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !sx2, index, index, index)

  // From lds_16x64_b.mlir
  func.func private @load_global_tile_16x64_b(!sx2, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @load_lds_A_swizzled(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled(index, index, index) -> !future_lds_read
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using triple-buffer LDS + AGPR
  amdgcn.kernel @gemm_16x16xK_lds_3buf arguments <[
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
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 32)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: triple buffer (1024 bytes per 16x32 tile)
    %lds_a0_h = amdgcn.alloc_lds 1024
    %lds_A0 = amdgcn.get_lds_offset %lds_a0_h : index
    %lds_b0_h = amdgcn.alloc_lds 1024
    %lds_B0 = amdgcn.get_lds_offset %lds_b0_h : index
    %lds_a1_h = amdgcn.alloc_lds 1024
    %lds_A1 = amdgcn.get_lds_offset %lds_a1_h : index
    %lds_b1_h = amdgcn.alloc_lds 1024
    %lds_B1 = amdgcn.get_lds_offset %lds_b1_h : index
    %lds_a2_h = amdgcn.alloc_lds 1024
    %lds_A2 = amdgcn.get_lds_offset %lds_a2_h : index
    %lds_b2_h = amdgcn.alloc_lds 1024
    %lds_B2 = amdgcn.get_lds_offset %lds_b2_h : index

    // Initialize accumulator to zero (AGPR)
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // === Prologue: prefetch iterations 0 and 1 ===
    // Prefetch iteration 0 into buffer 0
    %pf0_A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %c0, %stride_AB)
        : (!sx2, index, index, index) -> !future_global_read
    %pf0_B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %c0, %stride_AB)
        : (!sx2, index, index, index) -> !future_global_read
    %pf0_At0, %pf0_At1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A0, %pf0_A_gfut)
        : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
    %pf0_Bt0, %pf0_Bt1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B0, %pf0_B_gfut)
        : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
    amdgcn.wait deps %pf0_At0 : !lds_write_token
    amdgcn.wait deps %pf0_At1 : !lds_write_token
    amdgcn.wait deps %pf0_Bt0 : !lds_write_token
    amdgcn.wait deps %pf0_Bt1 : !lds_write_token

    // Prefetch iteration 1 into buffer 1 (if K_tiles > 1)
    %has_iter1 = arith.cmpi ugt, %K_tiles, %c1 : index
    scf.if %has_iter1 {
      %pf1_A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %c32, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read
      %pf1_B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %c32, %stride_AB)
          : (!sx2, index, index, index) -> !future_global_read
      %pf1_At0, %pf1_At1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A1, %pf1_A_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %pf1_Bt0, %pf1_Bt1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B1, %pf1_B_gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %pf1_At0 : !lds_write_token
      amdgcn.wait deps %pf1_At1 : !lds_write_token
      amdgcn.wait deps %pf1_Bt0 : !lds_write_token
      amdgcn.wait deps %pf1_Bt1 : !lds_write_token
    }

    // K-loop: triple-buffered iteration over K dimension
    %C_final, %buf_final = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(%acc = %C_init, %buf_idx = %c0) -> (!rt_C_f32, index) {
      // === Buffer Selection (3-way mux on buf_idx) ===
      %is_buf1 = arith.cmpi eq, %buf_idx, %c1 : index
      %lds_A_12 = arith.select %is_buf1, %lds_A1, %lds_A2 : index
      %lds_B_12 = arith.select %is_buf1, %lds_B1, %lds_B2 : index
      %is_buf0 = arith.cmpi eq, %buf_idx, %c0 : index
      %lds_A_cur = arith.select %is_buf0, %lds_A0, %lds_A_12 : index
      %lds_B_cur = arith.select %is_buf0, %lds_B0, %lds_B_12 : index

      // === Prefetch k+2 (if within bounds) ===
      %k_plus2 = arith.addi %k, %c2 : index
      %has_prefetch = arith.cmpi ult, %k_plus2, %K_tiles : index
      scf.if %has_prefetch {
        %pf_raw = arith.addi %buf_idx, %c2 : index
        %pf_ge3 = arith.cmpi uge, %pf_raw, %c3 : index
        %pf_wrapped = arith.subi %pf_raw, %c3 : index
        %pf_buf_idx = arith.select %pf_ge3, %pf_wrapped, %pf_raw : index
        %pf_is_buf1 = arith.cmpi eq, %pf_buf_idx, %c1 : index
        %pf_A_12 = arith.select %pf_is_buf1, %lds_A1, %lds_A2 : index
        %pf_B_12 = arith.select %pf_is_buf1, %lds_B1, %lds_B2 : index
        %pf_is_buf0 = arith.cmpi eq, %pf_buf_idx, %c0 : index
        %pf_A = arith.select %pf_is_buf0, %lds_A0, %pf_A_12 : index
        %pf_B = arith.select %pf_is_buf0, %lds_B0, %pf_B_12 : index
        %pf_offset = affine.apply affine_map<(k) -> (k * 32)>(%k_plus2)
        %pf_A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %c0, %pf_offset, %stride_AB)
            : (!sx2, index, index, index) -> !future_global_read
        %pf_B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %c0, %pf_offset, %stride_AB)
            : (!sx2, index, index, index) -> !future_global_read
        %pf_At0, %pf_At1 = func.call @store_global_tile_to_lds_16x64_b(%pf_A, %pf_A_gfut)
            : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
        %pf_Bt0, %pf_Bt1 = func.call @store_global_tile_to_lds_16x64_b(%pf_B, %pf_B_gfut)
            : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
        amdgcn.wait deps %pf_At0 : !lds_write_token
        amdgcn.wait deps %pf_At1 : !lds_write_token
        amdgcn.wait deps %pf_Bt0 : !lds_write_token
        amdgcn.wait deps %pf_Bt1 : !lds_write_token
      }

      // === K0 sub-tile (byte offset 0 within LDS row) ===
      %A0_future = func.call @load_lds_A_swizzled(%lds_A_cur, %c0, %c2) : (index, index, index) -> !future_lds_read
      %A0_tile = func.call @get_lds_read_value_vx2(%A0_future) : (!future_lds_read) -> !rt_A_f16
      %B0_future = func.call @load_lds_B_swizzled(%lds_B_cur, %c0, %c2) : (index, index, index) -> !future_lds_read
      %B0_tile = func.call @get_lds_read_value_vx2(%B0_future) : (!future_lds_read) -> !rt_B_f16
      %acc_k0 = func.call @mfma_f32_16x16x16_f16(%A0_tile, %B0_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // === K1 sub-tile (byte offset 32 within LDS row) ===
      %A1_future = func.call @load_lds_A_swizzled(%lds_A_cur, %c32, %c2) : (index, index, index) -> !future_lds_read
      %A1_tile = func.call @get_lds_read_value_vx2(%A1_future) : (!future_lds_read) -> !rt_A_f16
      %B1_future = func.call @load_lds_B_swizzled(%lds_B_cur, %c32, %c2) : (index, index, index) -> !future_lds_read
      %B1_tile = func.call @get_lds_read_value_vx2(%B1_future) : (!future_lds_read) -> !rt_B_f16
      %acc_k1 = func.call @mfma_f32_16x16x16_f16(%A1_tile, %B1_tile, %acc_k0)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Advance buffer index: (buf_idx + 1) % 3 via wrap
      %next_raw = arith.addi %buf_idx, %c1 : index
      %next_is_3 = arith.cmpi eq, %next_raw, %c3 : index
      %next_buf = arith.select %next_is_3, %c0, %next_raw : index

      scf.yield %acc_k1, %next_buf : !rt_C_f32, index
    }

    // Fire-and-forget store (s_endpgm drains outstanding stores)
    func.call @store_global_C_mfma_f32_16x16x16_f16(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
