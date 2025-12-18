// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/64/g' -e 's/{{SIZE_N}}/64/g' -e 's/{{SIZE_K}}/64/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_M}}/4/g' -e 's/{{LOOP_SIZE_N}}/4/g' -e 's/{{LOOP_SIZE_K}}/4/g' \
// RUN: | sed -e 's/{{SIZE_K_BY_TILE_SIZE_K}}/2/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_D_MMNNKK}}/6/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register_init.mlir,%p/library/common/indexing.mlir" \
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
  // register_init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @wave_id() -> index
  func.func private @wave_count() -> index
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @tiled_grid_partition_2D(index, index, index, index) -> (index, index)
  // copies.mlir
  func.func private @global_load_dwordx2_wait(
    !sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @lds_write_dwordx2_wait(
    index, index, index, index, index, !vx2) -> ()
  func.func private @read_lds_A_16x16xf16_fragment_wait(
    index, index, index, index) -> !vx2
  func.func private @store_global_16x16xf32_C_fragment_wait(
    !vx4, !sx2, index, index, index, index, index) -> ()

  // Phase 0a: Global loads if phase 0 (decoupled from DS writes via memrefs)
  func.func private @maybe_global_load(
    %phase: index, %k: index, %d_mmnnkk: index,
    %NN: index, %MM: index, %d_MMKK: index, %d_NNKK: index,
    %w: index, %W: index, %KK: index,
    %a_global: !sx2, %b_global: !sx2,
    %i_pos: index, %j_pos: index, %k_pos: index, %SIZE_K: index,
    %a_load_memref: memref<?x?x!vx2>, %b_load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %is_phase_0 = arith.cmpi eq, %phase, %c0 : index
    scf.if %is_phase_0 {
      %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
      scf.if %is_first_it {
        amdgcn.sopp.sopp <s_barrier>
      }

      // Global load A tile (decoupled: stores to memref)
      %_jj, %d_mmkk = affine.delinearize_index %d_mmnnkk into (%NN, %d_MMKK) : index, index
      %is_nn_zero = arith.cmpi eq, %_jj, %c0 : index
      scf.if %is_nn_zero {
        %iikk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_mmkk, %w, %W]
        %ii_pos = affine.apply affine_map<()[idx, KKv] -> (idx * (16 ceildiv KKv))>()[%iikk, %KK]
        %loaded = func.call @global_load_dwordx2_wait(%a_global, %i_pos, %k_pos, %SIZE_K, %ii_pos, %c0, %KK)
          : (!sx2, index, index, index, index, index, index) -> (!vx2)

        memref.store %loaded, %a_load_memref[%k, %d_mmkk] : memref<?x?x!vx2>
      }

      // Global load B tile (decoupled: stores to memref)
      %ii, %d_nnkk = affine.delinearize_index %d_mmnnkk into (%MM, %d_NNKK) : index, index
      %is_mm_zero = arith.cmpi eq, %ii, %c0 : index
      scf.if %is_mm_zero {
        %jjkk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_nnkk, %w, %W]
        %jj_pos = affine.apply affine_map<()[idx, KKv] -> (idx * (16 ceildiv KKv))>()[%jjkk, %KK]
        %loaded = func.call @global_load_dwordx2_wait(%b_global, %j_pos, %k_pos, %SIZE_K, %jj_pos, %c0, %KK)
          : (!sx2, index, index, index, index, index, index) -> (!vx2)

        memref.store %loaded, %b_load_memref[%k, %d_nnkk] : memref<?x?x!vx2>
      }
    }
    return
  }

  // Phase 0b: DS writes if phase 0 (decoupled from global loads via memrefs)
  func.func private @maybe_lds_write(
    %phase: index, %k: index, %d_mmnnkk: index,
    %NN: index, %MM: index, %d_MMKK: index, %d_NNKK: index,
    %w: index, %W: index, %KK: index,
    %lds_a_base_off: index, %lds_b_base_off: index, %TILE_SIZE_K: index,
    %a_load_memref: memref<?x?x!vx2>, %b_load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %is_phase_0 = arith.cmpi eq, %phase, %c0 : index
    scf.if %is_phase_0 {
      // DS write A tile (decoupled: reads from memref)
      %_jj, %d_mmkk = affine.delinearize_index %d_mmnnkk into (%NN, %d_MMKK) : index, index
      %is_nn_zero = arith.cmpi eq, %_jj, %c0 : index
      scf.if %is_nn_zero {
        %iikk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_mmkk, %w, %W]
        %ii_pos = affine.apply affine_map<()[idx, KKv] -> (idx * (16 ceildiv KKv))>()[%iikk, %KK]
        %loaded = memref.load %a_load_memref[%k, %d_mmkk] : memref<?x?x!vx2>
        func.call @lds_write_dwordx2_wait(%lds_a_base_off, %ii_pos, %c0, %TILE_SIZE_K, %KK, %loaded)
          : (index, index, index, index, index, !vx2) -> ()
      }

      // DS write B tile (decoupled: reads from memref)
      %ii, %d_nnkk = affine.delinearize_index %d_mmnnkk into (%MM, %d_NNKK) : index, index
      %is_mm_zero = arith.cmpi eq, %ii, %c0 : index
      scf.if %is_mm_zero {
        %jjkk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_nnkk, %w, %W]
        %jj_pos = affine.apply affine_map<()[idx, KKv] -> (idx * (16 ceildiv KKv))>()[%jjkk, %KK]
        %loaded = memref.load %b_load_memref[%k, %d_nnkk] : memref<?x?x!vx2>
        func.call @lds_write_dwordx2_wait(%lds_b_base_off, %jj_pos, %c0, %TILE_SIZE_K, %KK, %loaded)
          : (index, index, index, index, index, !vx2) -> ()
      }
    }
    return
  }

  // Phase 1a: LDS reads if phase 1 (decoupled from mfma via memrefs)
  func.func private @maybe_lds_read(
    %phase: index, %k: index, %d_mmnnkk: index, %d_MMNN: index, %KK: index,
    %w: index, %W: index, %MM: index, %NN: index,
    %lds_a_base_off: index, %lds_b_base_off: index, %TILE_SIZE_K: index,
    %a_frag_memref: memref<?x?x!vx2>, %b_frag_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_phase_1 = arith.cmpi eq, %phase, %c1 : index
    scf.if %is_phase_1 {
      %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
      scf.if %is_first_it {
        amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0 immutable
        amdgcn.sopp.sopp <s_barrier>
      }
      // Calculate mma tile indices
      %d_mmnn, %kk = affine.delinearize_index %d_mmnnkk into (%d_MMNN, %KK) : index, index
      %mmnn = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_mmnn, %w, %W]
      %ii, %jj = affine.delinearize_index %mmnn into (%MM, %NN) : index, index

      // Compute positions
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
      %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]

      // Read A and B fragments from LDS, store to memrefs
      %a_frag = func.call @read_lds_A_16x16xf16_fragment_wait(%lds_a_base_off, %ii_pos, %kk_pos, %TILE_SIZE_K)
        : (index, index, index, index) -> !vx2
      %b_frag = func.call @read_lds_A_16x16xf16_fragment_wait(%lds_b_base_off, %jj_pos, %kk_pos, %TILE_SIZE_K)
        : (index, index, index, index) -> !vx2
      memref.store %a_frag, %a_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>
      memref.store %b_frag, %b_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>
    }
    return
  }

  // Perform MFMA if phase 1: load fragments, compute, store result
  func.func private @maybe_mfma(
    %phase: index, %k: index, %d_mmnnkk: index, %d_MMNN: index, %KK: index,
    %a_frag_memref: memref<?x?x!vx2>, %b_frag_memref: memref<?x?x!vx2>,
    %c_fragments: memref<?x!vx4>
  ) {
    %c1 = arith.constant 1 : index
    %is_phase_1 = arith.cmpi eq, %phase, %c1 : index
    scf.if %is_phase_1 {
      // Load fragments from memrefs
      %a_frag = memref.load %a_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>
      %b_frag = memref.load %b_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>

      %d_mmnn, %kk = affine.delinearize_index %d_mmnnkk into (%d_MMNN, %KK) : index, index
      // Perform MFMA operation: C = A * B + C
      %acc = memref.load %c_fragments[%d_mmnn] : memref<?x!vx4>
      %updated_acc = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
        %acc, %a_frag, %b_frag, %acc : !vx2, !vx2, !vx4 -> !vx4
      memref.store %updated_acc, %c_fragments[%d_mmnn] : memref<?x!vx4>
    }
    return
  }

  // Store C fragment to global memory if phase 2 and at last k iteration
  func.func private @maybe_store_c_fragment(
    %phase: index,
    %d_mmnnkk: index, %d_MMNN: index, %KK: index,
    %w: index, %W: index, %MM: index, %NN: index,
    %k: index, %K: index,
    %i_pos: index, %j_pos: index, %SIZE_N: index,
    %c_fragments: memref<?x!vx4>, %c_global: !sx2
  ) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %is_phase_2 = arith.cmpi eq, %phase, %c2 : index
    scf.if %is_phase_2 {
      // Calculate mma tile indices
      %d_mmnn, %kk = affine.delinearize_index %d_mmnnkk into (%d_MMNN, %KK) : index, index
      %mmnn = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_mmnn, %w, %W]
      %ii, %jj = affine.delinearize_index %mmnn into (%MM, %NN) : index, index

      // if k is the last tile and kk is the last iteration, store to global
      %k_minus_1 = arith.subi %K, %c1 : index
      %kk_minus_1 = arith.subi %KK, %c1 : index
      %is_last_k = arith.cmpi eq, %k, %k_minus_1 : index
      %is_last_kk = arith.cmpi eq, %kk, %kk_minus_1 : index
      %is_last_k_and_kk = arith.andi %is_last_k, %is_last_kk : i1
      scf.if %is_last_k_and_kk {
        %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
        %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
        %fragment = memref.load %c_fragments[%d_mmnn] : memref<?x!vx4>
        func.call @store_global_16x16xf32_C_fragment_wait(%fragment, %c_global, %i_pos, %j_pos, %SIZE_N, %ii_pos, %jj_pos)
          : (!vx4, !sx2, index, index, index, index, index) -> ()
      }
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
    %w = func.call @wave_id() : () -> index
    %W = func.call @wave_count() : () -> index

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

    // Warp-level tile indices (ii, jj, kk) and sizes (MM, NN, KK)
    %MM = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_M]
    %NN = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_N]
    %KK = affine.apply affine_map<()[sz] -> (sz ceildiv 16)>()[%TILE_SIZE_K]

    // Number of distributed mma tiles
    %d_MMNNKK = affine.apply affine_map<()[sz0, sz1, sz2, W] -> ((sz0 * sz1 * sz2) ceildiv W)>()[%MM, %NN, %KK, %W]
    %d_MMNN = affine.apply affine_map<()[sz0, sz1, W] -> ((sz0 * sz1) ceildiv W)>()[%MM, %NN, %W]
    %d_MMKK = affine.apply affine_map<()[sz0, sz1, W] -> ((sz0 * sz1) ceildiv W)>()[%MM, %KK, %W]
    %d_NNKK = affine.apply affine_map<()[sz0, sz1, W] -> ((sz0 * sz1) ceildiv W)>()[%NN, %KK, %W]

    // Allocate registers for the C fragment
    %c_fragments = memref.alloca(%d_MMNN) : memref<?x!vx4>

    // Allocate memrefs for decoupled global loads -> DS writes
    %a_load_memref = memref.alloca(%K, %d_MMKK) : memref<?x?x!vx2>
    %b_load_memref = memref.alloca(%K, %d_NNKK) : memref<?x?x!vx2>

    // Allocate memrefs for decoupled LDS reads -> mfma
    %a_frag_memref = memref.alloca(%K, %d_MMNNKK) : memref<?x?x!vx2>
    %b_frag_memref = memref.alloca(%K, %d_MMNNKK) : memref<?x?x!vx2>

    // Initialize C fragments
    %c0_i32 = arith.constant 0 : i32
    scf.for %c = %c0 to %d_MMNN step %c1 {
      %c_fragment = func.call @init_vgprx4(%c0_i32) : (i32) -> !vx4
      memref.store %c_fragment, %c_fragments[%c] : memref<?x!vx4>
    } {amdgcn.constexpr}


    // M, N are fully distributed to blocks.
    // Loop over remaining 4-D tile **distributed** tile index (K, d_MMNNKK) in 2 phases:
    //   - Phase 0 loads to shared
    //   - Phase 1 computes
    //   - Phase 2 stores to global
    %num_phases = arith.constant 3 : index
    %ub = affine.apply affine_map<()[K, num_phases, d_MMNNKK] ->
         (K * num_phases * d_MMNNKK)>
      ()[%K, %num_phases, %d_MMNNKK]
    scf.for %idx = %c0 to %ub step %c1 {
      // Decompose linear index into 3D index
      %k, %phase, %d_mmnnkk = affine.delinearize_index %idx into (%K, %num_phases, %d_MMNNKK) : index, index, index
      %k_pos = affine.apply affine_map<(tile_size)[tile] -> (tile * tile_size)>(%TILE_SIZE_K)[%k]

      // Phase 0a: Global loads (decoupled from DS writes via memrefs)
      func.call @maybe_global_load(
        %phase, %k, %d_mmnnkk, %NN, %MM, %d_MMKK, %d_NNKK, %w, %W, %KK,
        %a_global, %b_global, %i_pos, %j_pos, %k_pos, %SIZE_K,
        %a_load_memref, %b_load_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (index, index, index, index, index, index, index, index, index, index,
           !sx2, !sx2, index, index, index, index,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 0b: DS writes (decoupled from global loads via memrefs)
      func.call @maybe_lds_write(
        %phase, %k, %d_mmnnkk, %NN, %MM, %d_MMKK, %d_NNKK, %w, %W, %KK,
        %lds_a_base_off, %lds_b_base_off, %TILE_SIZE_K,
        %a_load_memref, %b_load_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (index, index, index, index, index, index, index, index, index, index,
           index, index, index,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 1a: LDS reads (decoupled from mfma via memrefs)
      func.call @maybe_lds_read(
        %phase, %k, %d_mmnnkk, %d_MMNN, %KK, %w, %W, %MM, %NN,
        %lds_a_base_off, %lds_b_base_off, %TILE_SIZE_K,
        %a_frag_memref, %b_frag_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (index, index, index, index, index, index, index, index, index,
           index, index, index,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 1b: MFMA (decoupled from LDS reads via memrefs)
      func.call @maybe_mfma(
        %phase, %k, %d_mmnnkk, %d_MMNN, %KK,
        %a_frag_memref, %b_frag_memref, %c_fragments)
        {sched.delay = 6 : i64, sched.rate = 1 : i64}
        : (index, index, index, index, index,
           memref<?x?x!vx2>, memref<?x?x!vx2>, memref<?x!vx4>) -> ()

      // Phase 2: Store C fragment back to global memory
      func.call @maybe_store_c_fragment(
        %phase, %d_mmnnkk, %d_MMNN, %KK, %w, %W, %MM, %NN, %k, %K,
        %i_pos, %j_pos, %SIZE_N, %c_fragments, %c_global)
          {sched.delay = 12 : i64, sched.rate = 1 : i64}
        : (index, index, index, index, index, index, index, index, index, index,
           index, index, index, memref<?x!vx4>, !sx2) -> ()

    } {amdgcn.constexpr, sched.dims = array<i64: {{SIZE_K_BY_TILE_SIZE_K}}, 3, {{LOOP_SIZE_D_MMNNKK}}> }

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
