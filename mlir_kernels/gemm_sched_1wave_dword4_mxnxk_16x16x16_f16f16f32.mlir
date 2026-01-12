// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/64/g' -e 's/{{SIZE_N}}/64/g' -e 's/{{SIZE_K}}/64/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_M}}/4/g' -e 's/{{LOOP_SIZE_N}}/4/g' -e 's/{{LOOP_SIZE_K}}/4/g' \
// RUN: | sed -e 's/{{SIZE_K_BY_TILE_SIZE_K}}/2/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_D_MMNNKK}}/6/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

// CHECK-LABEL: amdgcn.module
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @wave_id() -> index
  func.func private @wave_count() -> index
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @tiled_grid_partition_2D(index, index, index, index) -> (index, index)
  // copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    index, index, index, index, index, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    index, index, index, index) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !sx2, index, index, index, index, index) -> ()
  // multi-tile-copies.mlir
  func.func private @maybe_global_load_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, !sx2, index, index, index, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)

  // Initialize C (decoupled via memrefs).
  func.func private @maybe_init_C(
    %k: index, %mm: index, %nn: index, %kk: index,               // indices
    %K: index, %MM: index, %NN: index,  %KK: index,              // sizes
    %c_fragments: memref<?x?x!vx4>                             // memref for decoupled global load
  ) {
    %c0 = arith.constant 0 : index
    %k_kk = affine.linearize_index [%k, %kk] by (%K, %KK) : index

    // Global load A tile (decoupled: stores to memref)
    %is_k_kk_zero = arith.cmpi eq, %k_kk, %c0 : index
    scf.if %is_k_kk_zero {
      %c0_i32 = arith.constant 0 : i32
      // %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
      %c_fragment = func.call @init_vgprx4(%c0_i32) : (i32) -> !vx4
      memref.store %c_fragment, %c_fragments[%mm, %nn] : memref<?x?x!vx4>
    }

    return
  }

  // Unified global load function (decoupled from DS writes via memrefs).
  // For A: call with (mm, kk, nn, MM, KK) → ii=mm, jj=kk, cond_iter=nn
  // For B: call with (nn, kk, mm, NN, KK) → ii=nn, jj=kk, cond_iter=mm
  // Executes load when cond_iter == 0, linearizes [ii, jj] by (II, JJ).
  func.func private @maybe_global_load(
    %k: index, %ii: index, %jj: index, %cond_iter: index,        // indices (cond_iter is checked for zero)
    %K: index, %II: index, %JJ: index,                           // sizes
    %ptr: !sx2,                                                  // global memory pointer
    %i_pos: index, %j_pos: index, %SIZE_J: index,                // global positions
    %load_memref: memref<?x?x?x!vx2>                             // memref for decoupled global load
  ) {
    %c0 = arith.constant 0 : index
    %iijj = affine.linearize_index [%ii, %jj] by (%II, %JJ) : index

    // Global load tile (decoupled: stores to memref)
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    scf.if %is_cond_zero {
      %num_rows = affine.apply affine_map<()[JJ] -> (16 ceildiv JJ)>()[%JJ]
      %ii_pos = affine.apply affine_map<()[iijj, num_rows] -> (iijj * num_rows)>()[%iijj, %num_rows]
      %elt_size = arith.constant 2 : index // f16 size in bytes
      %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_J, elt_size] ->
        (SIZE_J * elt_size)>()[%SIZE_J, %elt_size]
      %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(
          %ptr, %i_pos, %j_pos, %GLOBAL_STRIDE_IN_BYTES, %ii_pos, %c0, %num_rows)
        : (!sx2, index, index, index, index, index, index) -> (!vx2)
      memref.store %loaded, %load_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }

    return
  }

  // Unified LDS write function (decoupled from global loads via memrefs).
  // For A: call with (mm, kk, nn, MM, KK) → ii=mm, jj=kk, cond_iter=nn
  // For B: call with (nn, kk, mm, NN, KK) → ii=nn, jj=kk, cond_iter=mm
  // Executes write when cond_iter == 0, linearizes [ii, jj] by (II, JJ).
  func.func private @maybe_lds_write(
    %k: index, %ii: index, %jj: index, %cond_iter: index,  // indices (cond_iter is checked for zero)
    %K: index, %II: index, %JJ: index,                     // sizes
    %lds_base_off: index, %TILE_SIZE_K: index,             // base offset and stride
    %load_memref: memref<?x?x?x!vx2>                       // memref<reg> for decoupled LDS write
  ) {
    %c0 = arith.constant 0 : index
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    scf.if %is_cond_zero {
      %num_rows = affine.apply affine_map<()[JJ] -> (16 ceildiv JJ)>()[%JJ]
      %iijj = affine.linearize_index [%ii, %jj] by (%II, %JJ) : index
      %ii_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%iijj, %num_rows]
      %loaded = memref.load %load_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
      %elt_size = arith.constant 2 : index // f16 size in bytes
      %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[TILE_SIZE_K, elt_size] ->
        (TILE_SIZE_K * elt_size)>()[%TILE_SIZE_K, %elt_size]
      func.call @lds_write_wave_256xf16_via_dwordx2_wait(
          %lds_base_off, %ii_pos, %c0, %LDS_STRIDE_IN_BYTES, %num_rows, %loaded)
        : (index, index, index, index, index, !vx2) -> ()
    }
    return
  }

  // Unified LDS read function (decoupled from mfma via memrefs).
  // For A: ii=mm, jj=kk, cond_iter=nn → memref[k, ii, jj]
  // For B: ii=nn, jj=kk, cond_iter=mm → memref[k, ii, jj]
  // When cond_iter == 0: reads from LDS at (ii*16, jj*16), stores to memref
  // When cond_iter != 0: no-op (value already in memref from cond_iter==0)
  func.func private @maybe_lds_read(
    %k: index, %ii: index, %jj: index, %cond_iter: index,  // indices
    %lds_base_off: index, %TILE_SIZE_K: index,             // base offset and stride
    %frag_memref: memref<?x?x?x!vx2>                       // 3D memref for fragments
  ) {
    %c0 = arith.constant 0 : index
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index

    scf.if %is_cond_zero {
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
      %elt_size = arith.constant 2 : index // f16 size in bytes
      %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[TILE_SIZE_K, elt_size] ->
        (TILE_SIZE_K * elt_size)>()[%TILE_SIZE_K, %elt_size]
      // Note: LDS read A and B are the same function because B is NxK atm.
      %frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(
          %lds_base_off, %ii_pos, %jj_pos, %LDS_STRIDE_IN_BYTES)
        : (index, index, index, index) -> !vx2
      memref.store %frag, %frag_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }
    return
  }

  // Perform MFMA: load fragments, compute, store result fragment
  func.func private @maybe_mfma(
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %a_frag_memref: memref<?x?x?x!vx2>,    // register fragments for A
    %b_frag_memref: memref<?x?x?x!vx2>,    // register fragments for B
    %c_fragments: memref<?x?x!vx4>         // register fragments for C
  ) {
    %c1 = arith.constant 1 : index
    // Load fragments from memrefs
    %a_frag = memref.load %a_frag_memref[%k, %mm, %kk] : memref<?x?x?x!vx2>
    %b_frag = memref.load %b_frag_memref[%k, %nn, %kk] : memref<?x?x?x!vx2>
    // Perform MFMA operation: C = A * B + C
    %acc = memref.load %c_fragments[%mm, %nn] : memref<?x?x!vx4>
    %updated_acc = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
    %acc, %a_frag, %b_frag, %acc : !vx2, !vx2, !vx4 -> !vx4
    memref.store %updated_acc, %c_fragments[%mm, %nn] : memref<?x?x!vx4>
    return
  }

  // Store C fragment to global memory and at last k iteration
  func.func private @maybe_store_c_fragment(
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %c_fragments: memref<?x?x!vx4>,                   // register fragments for C
    %c_global: !sx2,                                // global memory pointer
    %m_pos: index, %n_pos: index, %SIZE_N: index    // global positions
  ) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    // if k is the last tile and kk is the last iteration, store to global
    %k_minus_1 = arith.subi %K, %c1 : index
    %kk_minus_1 = arith.subi %KK, %c1 : index
    %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
    %is_last_kk = arith.cmpi eq, %kk, %kk_minus_1 : index
    %is_last_k_and_kk = arith.andi %is_last_k, %is_last_kk : i1
    scf.if %is_last_k_and_kk {
      %fragment = memref.load %c_fragments[%mm, %nn] : memref<?x?x!vx4>
      %mm_pos = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
      %nn_pos = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]
      %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_N] ->
        (SIZE_N * 4)>()[%SIZE_N]
      func.call @global_store_wave_16x16xf32_C_fragment_wait(
          %fragment, %c_global, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos)
        : (!vx4, !sx2, index, index, index, index, index) -> ()
    }
    return
  }

  // Main function that allocates memrefs and loops over M, N, K
  func.func private @matmul_loop(
    %SIZE_M: index, %SIZE_N: index, %SIZE_K: index,            // Problem sizes
    %TILE_SIZE_M: index, %TILE_SIZE_N: index, %TILE_SIZE_K: index,   // Block-level tile sizes
    %a_global: !sx2, %b_global: !sx2, %c_global: !sx2 // Global memory pointers
  ) {
    // GPU variables
    %bdim = gpu.block_dim x
    %bid = gpu.block_id x
    %tid = gpu.thread_id x

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %elt_sz_in_b = arith.constant 2 : index

    // Calculate base indices in LDS for A and B
    %lds_a_base_off = arith.constant 0 : index
    %lds_b_base_off = affine.apply
      affine_map<()[rows, cols, type_size] -> (rows * cols * type_size)>
      ()[%TILE_SIZE_M, %TILE_SIZE_K, %elt_sz_in_b]

    // Block-level tile indices (i, j, k) and sizes (M, N, K)
    %K = affine.apply affine_map<()[sz, bsz] -> (sz ceildiv bsz)>()[%SIZE_K, %TILE_SIZE_K]
    %i, %j = func.call @tiled_grid_partition_2D(%SIZE_M, %SIZE_N, %TILE_SIZE_M, %TILE_SIZE_N)
      : (index, index, index, index) -> (index, index)

    // Calculate global positions
    %i_pos = affine.apply affine_map<(tile_size)[idx] -> (idx * tile_size)>(%TILE_SIZE_M)[%i]
    %j_pos = affine.apply affine_map<(tile_size)[idx] -> (idx * tile_size)>(%TILE_SIZE_N)[%j]
    %m_pos = affine.apply affine_map<(tile_size)[idx] -> (idx * tile_size)>(%TILE_SIZE_M)[%i]
    %n_pos = affine.apply affine_map<(tile_size)[idx] -> (idx * tile_size)>(%TILE_SIZE_N)[%j]

    // Warp-level tile indices (ii, jj, kk) and sizes (MM, NN, KK)
    %MM = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_M]
    %NN = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_N]
    %KK = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_K]

    // Multi-tile factors: load/write multiple tiles at once
    %NT_K = affine.min affine_map<()[KK] -> (4, KK)>()[%KK]
    %NT_M = affine.min affine_map<()[MM, NT_K] -> (4 ceildiv NT_K, MM)>()[%MM, %NT_K]
    %NT_N = affine.min affine_map<()[NN, NT_K] -> (4 ceildiv NT_K, NN)>()[%NN, %NT_K]

    // Allocate registers for the C fragment
    %c_fragments = memref.alloca(%MM, %NN) : memref<?x?x!vx4>

    // Allocate memrefs for decoupled global loads -> DS writes (2D: [K, NT_I*NT_J])
    %NT_MK = affine.apply affine_map<()[NT_M, NT_K] -> (NT_M * NT_K)>()[%NT_M, %NT_K]
    %NT_NK = affine.apply affine_map<()[NT_N, NT_K] -> (NT_N * NT_K)>()[%NT_N, %NT_K]
    %a_load_memref = memref.alloca(%K, %NT_MK) : memref<?x?x!vx2>
    %b_load_memref = memref.alloca(%K, %NT_NK) : memref<?x?x!vx2>

    // Allocate memrefs for decoupled LDS reads -> mfma
    %a_frag_memref = memref.alloca(%K, %MM, %KK) : memref<?x?x?x!vx2>
    %b_frag_memref = memref.alloca(%K, %NN, %KK) : memref<?x?x?x!vx2>

    // M, N are fully distributed to blocks.
    // Loop over remaining 4-D tile **distributed** tile index (K, MM, NN, KK)
    scf.for %k = %c0 to %K step %c1 {
      %ub = affine.apply affine_map<()[MM, NN, KK] ->
          (MM * NN * KK)>
        ()[%MM, %NN, %KK]
      scf.for %idx = %c0 to %ub step %c1 {
          %mm, %kk, %nn = affine.delinearize_index %idx into (%MM, %KK, %NN) : index, index, index
          %mmnnkk = affine.linearize_index [%mm, %nn, %kk] by (%MM, %NN, %KK) : index
          %k_pos = affine.apply affine_map<(tile_size)[tile] -> (tile * tile_size)>(%TILE_SIZE_K)[%k]

          // Multi-tile global load A: ii=mm, jj=kk, cond_iter=nn
          func.call @maybe_global_load_multi_tile_coalesced(
            %k, %mm, %kk, %nn,
            %K, %MM, %KK,
            %NT_M, %NT_K,
            %a_global, %m_pos, %k_pos, %SIZE_K, %a_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index,
              index, index,
              !sx2, index, index, index, memref<?x?x!vx2>) -> ()

          // Multi-tile global load B: ii=nn, jj=kk, cond_iter=mm
          func.call @maybe_global_load_multi_tile_coalesced(
            %k, %nn, %kk, %mm,
            %K, %NN, %KK,
            %NT_N, %NT_K,
            %b_global, %n_pos, %k_pos, %SIZE_K, %b_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index,
              index, index,
              !sx2, index, index, index, memref<?x?x!vx2>) -> ()

          // Multi-tile DS writes
          // Multi-tile DS write A: ii=mm, jj=kk, cond_iter=nn
          func.call @maybe_lds_write_multi_tile_coalesced(
            %k, %mm, %kk, %nn,
            %K, %MM, %KK,
            %NT_M, %NT_K,
            %lds_a_base_off, %TILE_SIZE_K, %a_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index,
              index, index,
              index, index, memref<?x?x!vx2>) -> ()

          // Multi-tile DS write B: ii=nn, jj=kk, cond_iter=mm
          func.call @maybe_lds_write_multi_tile_coalesced(
            %k, %nn, %kk, %mm,
            %K, %NN, %KK,
            %NT_N, %NT_K,
            %lds_b_base_off, %TILE_SIZE_K, %b_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index,
              index, index,
              index, index, memref<?x?x!vx2>) -> ()

          func.call @maybe_init_C(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %c_fragments)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index, index,
              memref<?x?x!vx4>) -> ()

          // LDS reads
          // LDS read A: ii=mm, jj=kk, cond_iter=nn
          func.call @maybe_lds_read(
            %k, %mm, %kk, %nn,
            %lds_a_base_off, %TILE_SIZE_K,
            %a_frag_memref)
              {sched.delay = 4 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // LDS read B: ii=nn, jj=kk, cond_iter=mm
          func.call @maybe_lds_read(
            %k, %nn, %kk, %mm,
            %lds_b_base_off, %TILE_SIZE_K,
            %b_frag_memref)
              {sched.delay = 4 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // MFMA (decoupled from LDS reads via memrefs)
          func.call @maybe_mfma(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %a_frag_memref, %b_frag_memref, %c_fragments)
              {sched.delay = 10 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index, index,
              memref<?x?x?x!vx2>, memref<?x?x?x!vx2>, memref<?x?x!vx4>) -> ()

          // Store C fragment back to global memory
          func.call @maybe_store_c_fragment(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %c_fragments, %c_global, %m_pos, %n_pos, %SIZE_N)
              {sched.delay = 12 : i64, sched.rate = 1 : i64}
          : (index, index, index, index,
            index, index, index, index,
            memref<?x?x!vx4>, !sx2, index, index, index) -> ()

      } {aster.constexpr, sched.dims = array<i64: {{LOOP_SIZE_D_MMNNKK}}> }
    } {aster.constexpr}

    return
  }

  //===--------------------------------------------------------------------===//
  // Kernel definition
  //===--------------------------------------------------------------------===//
  amdgcn.kernel @test_matmul_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = {{LDS_SIZE}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %a_ptr_s = amdgcn.load_arg 0 : !sx2
    %b_ptr_s = amdgcn.load_arg 1 : !sx2
    %c_ptr_s = amdgcn.load_arg 2 : !sx2
    %a_ptr, %b_ptr, %c_ptr = lsir.assume_noalias %a_ptr_s, %b_ptr_s, %c_ptr_s
      : (!sx2, !sx2, !sx2) -> (!sx2, !sx2, !sx2)

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // The problem sizes
    %SIZE_M = arith.constant {{SIZE_M}} : index
    %SIZE_N = arith.constant {{SIZE_N}} : index
    %SIZE_K = arith.constant {{SIZE_K}} : index
    %TILE_SIZE_M = arith.constant {{TILE_SIZE_M}} : index
    %TILE_SIZE_N = arith.constant {{TILE_SIZE_N}} : index
    %TILE_SIZE_K = arith.constant {{TILE_SIZE_K}} : index

    // Call the main matmul loop
    func.call @matmul_loop(%SIZE_M, %SIZE_N, %SIZE_K, %TILE_SIZE_M, %TILE_SIZE_N, %TILE_SIZE_K, %a_ptr, %b_ptr, %c_ptr)
      : (index, index, index, index, index, index, !sx2, !sx2, !sx2) -> ()

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0 lgkmcnt = 0

    amdgcn.end_kernel
  }
}
