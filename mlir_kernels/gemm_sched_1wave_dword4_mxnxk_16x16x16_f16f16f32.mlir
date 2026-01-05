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
  func.func private @global_load_64xdwordx2_wait(
    !sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @lds_write_64xdwordx2_wait(
    index, index, index, index, index, !vx2) -> ()
  func.func private @read_lds_A_16x16xf16_fragment_wait(
    index, index, index, index) -> !vx2
  func.func private @store_global_16x16xf32_C_fragment_wait(
    !vx4, !sx2, index, index, index, index, index) -> ()

  // Global load A (decoupled from DS writes via memrefs).
  func.func private @maybe_global_load_A(
    %d_mmnnkk: index,
    %k: index, %mm: index, %nn: index, %kk: index,               // indices
    %K: index, %MM: index, %NN: index,  %KK: index,              // sizes
    %a_global: !sx2,                                             // global memory pointers
    %m_pos: index, %n_pos: index, %k_pos: index, %SIZE_K: index, // global positions
    %a_load_memref: memref<?x?x?x!vx2>                             // memref for decoupled global load
  ) {
    %c0 = arith.constant 0 : index
    %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
    scf.if %is_first_it {
      amdgcn.sopp.sopp <s_barrier>
    }
    %mmkk = affine.linearize_index [%mm, %kk] by (%MM, %KK) : index

    // Global load A tile (decoupled: stores to memref)
    %is_nn_zero = arith.cmpi eq, %nn, %c0 : index
    scf.if %is_nn_zero {
      // %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
      %num_rows = affine.apply affine_map<()[] -> (16)>()[]
      %mm_pos = affine.apply affine_map<()[mm] -> (16 * mm)>()[%mm]
      %kk_pos = affine.apply affine_map<()[kk] -> (16 * kk)>()[%kk]
      %loaded = func.call @global_load_64xdwordx2_wait(%a_global, %m_pos, %k_pos, %SIZE_K, %mm_pos, %kk_pos, %num_rows)
        : (!sx2, index, index, index, index, index, index) -> (!vx2)
      memref.store %loaded, %a_load_memref[%k, %mm, %kk] : memref<?x?x?x!vx2>
    }

    return
  }

  // Global load B (decoupled from DS writes via memrefs).
  // Note: B is transposed (i.e. has layout NNxKK)
  func.func private @maybe_global_load_B(
    %d_mmnnkk: index,
    %k: index, %mm: index, %nn: index, %kk: index,               // indices
    %K: index, %MM: index, %NN: index,  %KK: index,              // sizes
    %b_global: !sx2,                                             // global memory pointers
    %m_pos: index, %n_pos: index, %k_pos: index, %SIZE_K: index, // global positions
    %b_load_memref: memref<?x?x?x!vx2>                             // memref for decoupled global
  ) {
    %c0 = arith.constant 0 : index
    %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
    scf.if %is_first_it {
      amdgcn.sopp.sopp <s_barrier>
    }
    %nnkk = affine.linearize_index [%nn, %kk] by (%NN, %KK) : index

    %is_mm_zero = arith.cmpi eq, %mm, %c0 : index
    scf.if %is_mm_zero {
      // %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
      %num_rows = affine.apply affine_map<()[] -> (16)>()[]
      %nn_pos = affine.apply affine_map<()[nn] -> (16 * nn)>()[%nn]
      %kk_pos = affine.apply affine_map<()[kk] -> (16 * kk)>()[%kk]
      %loaded = func.call @global_load_64xdwordx2_wait(%b_global, %n_pos, %k_pos, %SIZE_K, %nn_pos, %kk_pos, %num_rows)
        : (!sx2, index, index, index, index, index, index) -> (!vx2)
      memref.store %loaded, %b_load_memref[%k, %nn, %kk] : memref<?x?x?x!vx2>
    }

    return
  }

  // DS write A (decoupled from global loads via memrefs)
  func.func private @maybe_lds_write_A(
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %lds_a_base_off: index, %TILE_SIZE_K: index,    // base offset and stride for A
    %a_load_memref: memref<?x?x?x!vx2>                // memref<reg> for decoupled LDS write
  ) {
    %c0 = arith.constant 0 : index
    %is_nn_zero = arith.cmpi eq, %nn, %c0 : index
    scf.if %is_nn_zero {
      // %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
      %num_rows = affine.apply affine_map<()[] -> (16)>()[]
      %mm_pos = affine.apply affine_map<()[mm] -> (16 * mm)>()[%mm]
      %kk_pos = affine.apply affine_map<()[kk] -> (16 * kk)>()[%kk]
      %loaded = memref.load %a_load_memref[%k, %mm, %kk] : memref<?x?x?x!vx2>
      func.call @lds_write_64xdwordx2_wait(%lds_a_base_off, %mm_pos, %kk_pos, %TILE_SIZE_K, %num_rows, %loaded)
        : (index, index, index, index, index, !vx2) -> ()
    }
    return
  }

  // DS write B (decoupled from global loads via memrefs)
  // Note: B is transposed (i.e. has layout NNxKK)
  func.func private @maybe_lds_write_B(
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %lds_b_base_off: index, %TILE_SIZE_K: index,    // base offset and stride for B
    %b_load_memref: memref<?x?x?x!vx2>                // memref<reg> for decoupled LDS write
  ) {
    %c0 = arith.constant 0 : index
    %is_mm_zero = arith.cmpi eq, %mm, %c0 : index
    scf.if %is_mm_zero {
      // %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
      %num_rows = affine.apply affine_map<()[] -> (16)>()[]
      %nn_pos = affine.apply affine_map<()[nn] -> (16 * nn)>()[%nn]
      %kk_pos = affine.apply affine_map<()[kk] -> (16 * kk)>()[%kk]
      %loaded = memref.load %b_load_memref[%k, %nn, %kk] : memref<?x?x?x!vx2>
      func.call @lds_write_64xdwordx2_wait(%lds_b_base_off, %nn_pos, %kk_pos, %TILE_SIZE_K, %num_rows, %loaded)
        : (index, index, index, index, index, !vx2) -> ()
    }
    return
  }
  // LDS read A (decoupled from mfma via memrefs)
  func.func private @maybe_lds_read_A(
    %d_mmnnkk: index,
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %lds_a_base_off: index, %TILE_SIZE_K: index,    // base offset and stride for A
    %a_frag_memref: memref<?x?x?x?x!vx2>                // memref<reg> for decoupled LDS read
  ) {
    %c0 = arith.constant 0 : index
    %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
    scf.if %is_first_it {
      amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0 immutable
      amdgcn.sopp.sopp <s_barrier>
    }

    %is_nn_zero = arith.cmpi eq, %nn, %c0 : index
    scf.if %is_nn_zero {
      %mm_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%mm]
      %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]
      %a_frag = func.call @read_lds_A_16x16xf16_fragment_wait(%lds_a_base_off, %mm_pos, %kk_pos, %TILE_SIZE_K)
        : (index, index, index, index) -> !vx2
      %mmnnkk = affine.linearize_index [%mm, %nn, %kk] by (%MM, %NN, %KK) : index
      memref.store %a_frag, %a_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
    } else {
      %a_frag = memref.load %a_frag_memref[%k, %mm, %c0, %kk] : memref<?x?x?x?x!vx2>
      memref.store %a_frag, %a_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
    }
    return
  }

  // LDS read B (decoupled from mfma via memrefs)
  // Note: B is transposed (i.e. has layout NNxKK)
  func.func private @maybe_lds_read_B(
%d_mmnnkk: index,
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %lds_b_base_off: index, %TILE_SIZE_K: index,    // base offset and stride for B
    %b_frag_memref: memref<?x?x?x?x!vx2>                // memref<reg> for decoupled LDS read
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %is_first_it = arith.cmpi eq, %d_mmnnkk, %c0 : index
    scf.if %is_first_it {
      amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0 immutable
      amdgcn.sopp.sopp <s_barrier>
    }
    %is_mm_zero = arith.cmpi eq, %mm, %c0 : index
    scf.if %is_mm_zero {
      %nn_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%nn]
      %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]
      %b_frag = func.call @read_lds_A_16x16xf16_fragment_wait(%lds_b_base_off, %nn_pos, %kk_pos, %TILE_SIZE_K)
        : (index, index, index, index) -> !vx2
      %mmnnkk = affine.linearize_index [%mm, %nn, %kk] by (%MM, %NN, %KK) : index
      memref.store %b_frag, %b_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
    } else {
      %b_frag = memref.load %b_frag_memref[%k, %c0, %nn, %kk] : memref<?x?x?x?x!vx2>
      memref.store %b_frag, %b_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
    }
    return
  }

  // Perform MFMA: load fragments, compute, store result fragment
  func.func private @maybe_mfma(
    %k: index, %mm: index, %nn: index, %kk: index,  // indices
    %K: index, %MM: index, %NN: index,  %KK: index, // sizes
    %a_frag_memref: memref<?x?x?x?x!vx2>,    // register fragments for A
    %b_frag_memref: memref<?x?x?x?x!vx2>,    // register fragments for B
    %c_fragments: memref<?x?x!vx4>         // register fragments for C
  ) {
    %c1 = arith.constant 1 : index
    // Load fragments from memrefs
    %a_frag = memref.load %a_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
    %b_frag = memref.load %b_frag_memref[%k, %mm, %nn, %kk] : memref<?x?x?x?x!vx2>
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
      func.call @store_global_16x16xf32_C_fragment_wait(%fragment, %c_global, %m_pos, %n_pos, %SIZE_N, %mm_pos, %nn_pos)
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

    // Allocate registers for the C fragment
    %c_fragments = memref.alloca(%MM, %NN) : memref<?x?x!vx4>

    // Allocate memrefs for decoupled global loads -> DS writes
    %a_load_memref = memref.alloca(%K, %MM, %KK) : memref<?x?x?x!vx2>
    %b_load_memref = memref.alloca(%K, %NN, %KK) : memref<?x?x?x!vx2>

    // Allocate memrefs for decoupled LDS reads -> mfma
    %a_frag_memref = memref.alloca(%K, %MM, %NN, %KK) : memref<?x?x?x?x!vx2>
    %b_frag_memref = memref.alloca(%K, %MM, %NN, %KK) : memref<?x?x?x?x!vx2>

    // Initialize C fragments
    %c0_i32 = arith.constant 0 : i32
    scf.for %m = %c0 to %MM step %c1 {
      scf.for %n = %c0 to %NN step %c1 {
        %c_fragment = func.call @init_vgprx4(%c0_i32) : (i32) -> !vx4
        memref.store %c_fragment, %c_fragments[%m, %n] : memref<?x?x!vx4>
      } {amdgcn.constexpr}
    } {amdgcn.constexpr}

    // M, N are fully distributed to blocks.
    // Loop over remaining 4-D tile **distributed** tile index (K, MM, NN, KK)
    scf.for %k = %c0 to %K step %c1 {
      %ub = affine.apply affine_map<()[MM, NN, KK] ->
          (MM * NN * KK)>
        ()[%MM, %NN, %KK]
      scf.for %idx = %c0 to %ub step %c1 {
          // mmkknn order
          %mm, %kk, %nn = affine.delinearize_index %idx into (%MM, %KK, %NN) : index, index, index
          %mmnnkk = affine.linearize_index [%mm, %nn, %kk] by (%MM, %NN, %KK) : index
          %k_pos = affine.apply affine_map<(tile_size)[tile] -> (tile * tile_size)>(%TILE_SIZE_K)[%k]

          func.call @maybe_global_load_A(
            %mmnnkk,
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %a_global,
            %m_pos, %n_pos, %k_pos, %SIZE_K,
            %a_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (index,
              index, index, index, index,
              index, index, index, index,
              !sx2, index, index, index, index,
              memref<?x?x?x!vx2>) -> ()

          // Global load B (decoupled from DS writes via memrefs)
          func.call @maybe_global_load_B(
            %mmnnkk,
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %b_global,
            %m_pos, %n_pos, %k_pos, %SIZE_K,
            %b_load_memref)
              {sched.delay = 5 : i64, sched.rate = 1 : i64}
            : (index,
              index, index, index, index,
              index, index, index, index,
              !sx2, index, index, index, index,
              memref<?x?x?x!vx2>) -> ()

          // Phase 0b: DS writes
          // DS writeA (decoupled from global loads via memrefs)
          func.call @maybe_lds_write_A(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %lds_a_base_off, %TILE_SIZE_K,
            %a_load_memref)
              {sched.delay = 10 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index, index,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // DS writeB (decoupled from global loads via memrefs)
          func.call @maybe_lds_write_B(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %lds_b_base_off, %TILE_SIZE_K,
            %b_load_memref)
              {sched.delay = 15 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index, index,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // Phase 1a: LDS reads
          // DS readA (decoupled from global loads via memrefs)
          func.call @maybe_lds_read_A(
            %mmnnkk,
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %lds_a_base_off, %TILE_SIZE_K,
            %a_frag_memref)
              {sched.delay = 20 : i64, sched.rate = 1 : i64}
            : (index,
              index, index, index, index,
              index, index, index, index,
              index, index,
              memref<?x?x?x?x!vx2>) -> ()

          // DS readB (decoupled from global loads via memrefs)
          func.call @maybe_lds_read_B(
            %mmnnkk,
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %lds_b_base_off, %TILE_SIZE_K,
            %b_frag_memref)
              {sched.delay = 25 : i64, sched.rate = 1 : i64}
            : (index,
              index, index, index, index,
              index, index, index, index,
              index, index,
              memref<?x?x?x?x!vx2>) -> ()

          // Phase 1b: MFMA (decoupled from LDS reads via memrefs)
          func.call @maybe_mfma(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %a_frag_memref, %b_frag_memref, %c_fragments)
              {sched.delay = 50 : i64, sched.rate = 1 : i64}
            : (index, index, index, index,
              index, index, index, index,
              memref<?x?x?x?x!vx2>, memref<?x?x?x?x!vx2>, memref<?x?x!vx4>) -> ()

          // Phase 2: Store C fragment back to global memory
          func.call @maybe_store_c_fragment(
            %k, %mm, %nn, %kk,
            %K, %MM, %NN, %KK,
            %c_fragments,
            %c_global,
            %m_pos, %n_pos, %SIZE_N)
              {sched.delay = 100 : i64, sched.rate = 1 : i64}
          : (index, index, index, index,
            index, index, index, index,
            memref<?x?x!vx4>,
            !sx2,
            index, index, index) -> ()

      } {amdgcn.constexpr, sched.dims = array<i64: {{LOOP_SIZE_D_MMNNKK}}> }
    } {amdgcn.constexpr}

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
