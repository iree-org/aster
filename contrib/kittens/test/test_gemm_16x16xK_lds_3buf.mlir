// Kittens GEMM kernel with triple-buffer LDS: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Triple buffering adds a third buffer beyond double-buffering:
// - Buffer k%3:     being consumed (LDS -> Register -> Compute)
// - Buffer (k+1)%3: previous prefetch completing
// - Buffer (k+2)%3: starting new prefetch
//
// This eliminates barrier stalls when compute is faster than memory,
// giving loads an extra iteration of slack to complete.
//
// Template parameters:
//   {{K}}         - K dimension (e.g., 48, 64, 96)
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

amdgcn.module @kittens_gemm_16x16xK_lds_3buf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16.mlir
  func.func private @alloc_lds_3buffer() -> (index, index, index, index, index, index)
  func.func private @lds_barrier()

  // From kittens/lds_transfers.mlir
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_f16(index) -> !rt_B_f16

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using triple-buffer LDS
  //
  // Memory flow (steady state):
  //   Cycle 0:    Prefetch k+2: Global -> LDS[buf (k+2)%3] (async, 400 cycles)
  //   Cycle 0:    Wait for k's data in LDS[buf k%3]
  //   Cycle 10:   Load k: LDS[buf k%3] -> Register (~10 cycles)
  //   Cycle 20:   Compute k: MFMA (~16 cycles)
  //   Cycle 36:   Next iteration
  //   (Background: k+2 load has 2 full iterations (~72 cycles) of slack)
  amdgcn.kernel @gemm_16x16xK_lds_3buf arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 3264 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: triple buffer for A and B tiles
    // A[0]: offset 0,    B[0]: offset 544   (buffer 0)
    // A[1]: offset 1088, B[1]: offset 1632  (buffer 1)
    // A[2]: offset 2176, B[2]: offset 2720  (buffer 2)
    %lds_A0, %lds_B0, %lds_A1, %lds_B1, %lds_A2, %lds_B2 = func.call @alloc_lds_3buffer()
        : () -> (index, index, index, index, index, index)

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // === Prologue: prefetch iterations 0 and 1 ===
    // Prefetch iteration 0 into buffer 0
    func.call @load_global_to_lds_f16(%lds_A0, %A_ptr, %c0, %c0, %stride_AB)
        : (index, !sx2, index, index, index) -> ()
    func.call @load_global_to_lds_f16(%lds_B0, %B_ptr, %c0, %c0, %stride_AB)
        : (index, !sx2, index, index, index) -> ()

    // Prefetch iteration 1 into buffer 1 (if K_tiles > 1)
    %has_iter1 = arith.cmpi ugt, %K_tiles, %c1 : index
    scf.if %has_iter1 {
      func.call @load_global_to_lds_f16(%lds_A1, %A_ptr, %c0, %c16, %stride_AB)
          : (index, !sx2, index, index, index) -> ()
      func.call @load_global_to_lds_f16(%lds_B1, %B_ptr, %c0, %c16, %stride_AB)
          : (index, !sx2, index, index, index) -> ()
    }

    // K-loop: triple-buffered iteration over K dimension
    // buf_idx cycles 0->1->2->0->... as an iter_arg (avoids remui which is NYI for non-powers of 2)
    %C_final, %buf_final = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(%acc = %C_init, %buf_idx = %c0) -> (!rt_C_f32, index) {
      // === Buffer Selection (3-way mux on buf_idx) ===
      // Each cmpi must be consumed by its select before the next cmpi,
      // because i1 values live in SCC and cannot overlap.
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
        // Prefetch target buffer: (buf_idx + 2) % 3
        // (using select to avoid remui which is NYI for non-powers of 2)
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

        // Compute column offset for iteration k+2
        %pf_offset = arith.muli %k_plus2, %c16 : index

        // Async load k+2 into prefetch buffer (runs in background)
        func.call @load_global_to_lds_f16(%pf_A, %A_ptr, %c0, %pf_offset, %stride_AB)
            : (index, !sx2, index, index, index) -> ()
        func.call @load_global_to_lds_f16(%pf_B, %B_ptr, %c0, %pf_offset, %stride_AB)
            : (index, !sx2, index, index, index) -> ()
      }

      // === Wait for current iteration k's data ===
      // Barrier: ensure all threads completed their LDS writes for iteration k
      func.call @lds_barrier() : () -> ()

      // === Load LDS -> Register ===
      // Each thread loads its portion from current LDS buffer
      %A_tile = func.call @load_lds_to_register_A_f16(%lds_A_cur)
          : (index) -> !rt_A_f16
      %B_tile = func.call @load_lds_to_register_B_f16(%lds_B_cur)
          : (index) -> !rt_B_f16

      // === Compute ===
      // MFMA: acc += A_tile @ B_tile^T
      // While this computes, k+2's data loads in background (if has_prefetch)
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Advance buffer index: (buf_idx + 1) % 3 via wrap
      // (using select to avoid remui which is NYI for non-powers of 2)
      %next_raw = arith.addi %buf_idx, %c1 : index
      %next_is_3 = arith.cmpi eq, %next_raw, %c3 : index
      %next_buf = arith.select %next_is_3, %c0, %next_raw : index

      scf.yield %new_acc, %next_buf : !rt_C_f32, index
    }

    // Store result to global memory
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
