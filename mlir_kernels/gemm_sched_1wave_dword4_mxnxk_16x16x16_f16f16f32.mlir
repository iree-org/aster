// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/64/g' -e 's/{{SIZE_N}}/64/g' -e 's/{{SIZE_K}}/64/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_M}}/4/g' -e 's/{{LOOP_SIZE_N}}/4/g' -e 's/{{LOOP_SIZE_K}}/4/g' \
// RUN: | sed -e 's/{{SIZE_K_BY_TILE_SIZE_K}}/2/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_D_MMNNKK}}/6/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/descriptors.mlir" \
// RUN: | FileCheck %s

// CHECK-LABEL: amdgcn.module

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!index_pair = !aster_utils.struct<i: index, j: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

// A 4D tile index descriptor containing:
//   - k: outer K tile index
//   - mm, nn, kk: inner tile indices in M, N, K dimensions
!tile_index_descriptor_4d = !aster_utils.struct<k: index, mm: index, nn: index, kk: index>

// A 4D tile dimensions descriptor containing:
//   - K, MM, NN, KK: tile counts in each dimension
!tile_dims_descriptor_4d = !aster_utils.struct<K: index, MM: index, NN: index, KK: index>

amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4
  // From indexing.mlir
  func.func private @tiled_grid_partition_2d(!index_pair, !index_pair) -> !index_pair
  // From copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    !lds_position_descriptor_2d, i1) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
  // From conditional-multi-tile-copies.mlir
  func.func private @maybe_global_load_multi_tile_coalesced(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)

  // Initialize C (decoupled via memrefs).
  func.func private @maybe_init_C(
    %idx_desc: !tile_index_descriptor_4d,
    %dims_desc: !tile_dims_descriptor_4d,
    %c_fragments: memref<?x?x!vx4>
  ) {
    %c0 = arith.constant 0 : index
    // Extract indices
    %k = aster_utils.struct_extract %idx_desc["k"] : !tile_index_descriptor_4d -> index
    %mm = aster_utils.struct_extract %idx_desc["mm"] : !tile_index_descriptor_4d -> index
    %nn = aster_utils.struct_extract %idx_desc["nn"] : !tile_index_descriptor_4d -> index
    %kk = aster_utils.struct_extract %idx_desc["kk"] : !tile_index_descriptor_4d -> index
    // Extract dimensions
    %K = aster_utils.struct_extract %dims_desc["K"] : !tile_dims_descriptor_4d -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !tile_dims_descriptor_4d -> index

    %k_kk = affine.linearize_index [%k, %kk] by (%K, %KK) : index

    // Initialize C fragment at first k iteration
    %is_k_kk_zero = arith.cmpi eq, %k_kk, %c0 : index
    scf.if %is_k_kk_zero {
      %c0_i32 = arith.constant 0 : i32
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
    %cond_desc: !conditional_execution_descriptor_2d,           // k, cond_iter, NT_I=II, NT_J=JJ
    %pos_desc_base: !tensor_position_descriptor_2d,                // ptr, i_pos, j_pos, global_stride_in_bytes, elt_size
    %ii: index, %jj: index,                                      // inner tile indices for linearization and memref indexing
    %load_memref: memref<?x?x?x!vx2>                             // memref for decoupled global load
  ) {
    // Extract from conditional execution descriptor
    %k, %cond_iter, %II, %JJ = aster_utils.struct_extract %cond_desc ["k", "cond_iter", "NT_I", "NT_J"] : !conditional_execution_descriptor_2d -> index, index, index, index

    // Extract from position descriptor
    %ptr, %i_pos, %j_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc_base ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "elt_size"] : !tensor_position_descriptor_2d -> !sx2, index, index, index, index

    %c0 = arith.constant 0 : index
    %iijj = affine.linearize_index [%ii, %jj] by (%II, %JJ) : index

    // Global load tile (decoupled: stores to memref)
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    scf.if %is_cond_zero {
      %num_rows = affine.apply affine_map<()[JJ] -> (16 ceildiv JJ)>()[%JJ]
      %ii_pos = affine.apply affine_map<()[iijj, num_rows] -> (iijj * num_rows)>()[%iijj, %num_rows]
      %pos_desc = aster_utils.struct_create(%ptr, %i_pos, %j_pos, %GLOBAL_STRIDE_IN_BYTES, %ii_pos, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
      %transfer_size = arith.constant 8 : index // dwordx2
      %wave_size = arith.constant 64 : index
      %transfer_desc = aster_utils.struct_create(%num_rows, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
      %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
      memref.store %loaded, %load_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }

    return
  }

  // Unified LDS write function (decoupled from global loads via memrefs).
  // For A: call with (mm, kk, nn, MM, KK) → ii=mm, jj=kk, cond_iter=nn
  // For B: call with (nn, kk, mm, NN, KK) → ii=nn, jj=kk, cond_iter=mm
  // Executes write when cond_iter == 0, linearizes [ii, jj] by (II, JJ).
  func.func private @maybe_lds_write(
    %cond_desc: !conditional_execution_descriptor_2d,           // k, cond_iter, NT_I=II, NT_J=JJ
    %lds_pos_desc_base: !lds_position_descriptor_2d,            // lds_base, m_pos (unused), n_pos (unused), lds_stride_in_bytes, elt_size
    %ii: index, %jj: index,                                    // inner tile indices for linearization and memref indexing
    %load_memref: memref<?x?x?x!vx2>                          // memref<reg> for decoupled LDS write
  ) {
    // Extract from conditional execution descriptor
    %k, %cond_iter, %II, %JJ = aster_utils.struct_extract %cond_desc ["k", "cond_iter", "NT_I", "NT_J"] : !conditional_execution_descriptor_2d -> index, index, index, index

    // Extract from LDS position descriptor base
    %lds_base_off, %unused_m_pos, %unused_n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %lds_pos_desc_base ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index

    %c0 = arith.constant 0 : index
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index
    scf.if %is_cond_zero {
      %num_rows = affine.apply affine_map<()[JJ] -> (16 ceildiv JJ)>()[%JJ]
      %iijj = affine.linearize_index [%ii, %jj] by (%II, %JJ) : index
      %ii_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%iijj, %num_rows]
      %loaded = memref.load %load_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
      %lds_pos_desc = aster_utils.struct_create(%lds_base_off, %ii_pos, %c0, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
      %transfer_size_lds = arith.constant 8 : index // dwordx2
      %wave_size_lds = arith.constant 64 : index
      %transfer_desc_lds = aster_utils.struct_create(%num_rows, %transfer_size_lds, %wave_size_lds) : (index, index, index) -> !transfer_descriptor_2d
      func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc, %transfer_desc_lds, %loaded)
        : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
    }
    return
  }

  // Unified LDS read function (decoupled from mfma via memrefs).
  // For A: ii=mm, jj=kk, cond_iter=nn → memref[k, ii, jj]
  // For B: ii=nn, jj=kk, cond_iter=mm → memref[k, ii, jj]
  // When cond_iter == 0: reads from LDS at (ii*16, jj*16), stores to memref
  // When cond_iter != 0: no-op (value already in memref from cond_iter==0)
  func.func private @maybe_lds_read(
    %cond_desc: !conditional_execution_descriptor_2d,           // k, cond_iter, NT_I (unused), NT_J (unused)
    %lds_pos_desc_base: !lds_position_descriptor_2d,           // lds_base, m_pos (unused), n_pos (unused), lds_stride_in_bytes, elt_size
    %ii: index, %jj: index,                                   // inner tile indices for position computation and memref indexing
    %frag_memref: memref<?x?x?x!vx2>                         // 3D memref for fragments
  ) {
    // Extract from conditional execution descriptor
    %k, %cond_iter, %unused_NT_I, %unused_NT_J = aster_utils.struct_extract %cond_desc ["k", "cond_iter", "NT_I", "NT_J"] : !conditional_execution_descriptor_2d -> index, index, index, index

    // Extract from LDS position descriptor base
    %lds_base_off, %unused_m_pos, %unused_n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %lds_pos_desc_base ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index

    %c0 = arith.constant 0 : index
    %is_cond_zero = arith.cmpi eq, %cond_iter, %c0 : index

    scf.if %is_cond_zero {
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
      // Note: LDS read A and B are the same function because B is NxK atm.
      %false = arith.constant false
      %lds_pos_desc_read = aster_utils.struct_create(%lds_base_off, %ii_pos, %jj_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
      %frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc_read, %false)
        : (!lds_position_descriptor_2d, i1) -> !vx2
      memref.store %frag, %frag_memref[%k, %ii, %jj] : memref<?x?x?x!vx2>
    }
    return
  }

  // Perform MFMA: load fragments, compute, store result fragment
  func.func private @maybe_mfma(
    %idx_desc: !tile_index_descriptor_4d,
    %a_frag_memref: memref<?x?x?x!vx2>,
    %b_frag_memref: memref<?x?x?x!vx2>,
    %c_fragments: memref<?x?x!vx4>
  ) {
    // Extract indices
    %k = aster_utils.struct_extract %idx_desc["k"] : !tile_index_descriptor_4d -> index
    %mm = aster_utils.struct_extract %idx_desc["mm"] : !tile_index_descriptor_4d -> index
    %nn = aster_utils.struct_extract %idx_desc["nn"] : !tile_index_descriptor_4d -> index
    %kk = aster_utils.struct_extract %idx_desc["kk"] : !tile_index_descriptor_4d -> index

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

  // Store C fragment to global memory at last k iteration
  func.func private @maybe_store_c_fragment(
    %idx_desc: !tile_index_descriptor_4d,
    %dims_desc: !tile_dims_descriptor_4d,
    %c_fragments: memref<?x?x!vx4>,
    %tensor_desc_c: !tensor_position_descriptor_2d
  ) {
    %c1 = arith.constant 1 : index
    // Extract indices
    %k = aster_utils.struct_extract %idx_desc["k"] : !tile_index_descriptor_4d -> index
    %mm = aster_utils.struct_extract %idx_desc["mm"] : !tile_index_descriptor_4d -> index
    %nn = aster_utils.struct_extract %idx_desc["nn"] : !tile_index_descriptor_4d -> index
    %kk = aster_utils.struct_extract %idx_desc["kk"] : !tile_index_descriptor_4d -> index
    // Extract dimensions
    %K = aster_utils.struct_extract %dims_desc["K"] : !tile_dims_descriptor_4d -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !tile_dims_descriptor_4d -> index
    // Extract tensor info
    %c_global = aster_utils.struct_extract %tensor_desc_c["ptr"] : !tensor_position_descriptor_2d -> !sx2
    %m_pos = aster_utils.struct_extract %tensor_desc_c["m_pos"] : !tensor_position_descriptor_2d -> index
    %n_pos = aster_utils.struct_extract %tensor_desc_c["n_pos"] : !tensor_position_descriptor_2d -> index
    %GLOBAL_STRIDE_IN_BYTES = aster_utils.struct_extract %tensor_desc_c["global_stride_in_bytes"] : !tensor_position_descriptor_2d -> index
    %elt_size_c = aster_utils.struct_extract %tensor_desc_c["elt_size"] : !tensor_position_descriptor_2d -> index

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
      %pos_desc_c = aster_utils.struct_create(%c_global, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size_c) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
      %true_store = arith.constant true
      func.call @global_store_wave_16x16xf32_C_fragment_wait(%fragment, %pos_desc_c, %true_store) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
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
    %sizes = aster_utils.struct_create(%SIZE_M, %SIZE_N) : (index, index) -> !index_pair
    %tile_sizes = aster_utils.struct_create(%TILE_SIZE_M, %TILE_SIZE_N) : (index, index) -> !index_pair
    %result = func.call @tiled_grid_partition_2d(%sizes, %tile_sizes) : (!index_pair, !index_pair) -> !index_pair
    %i, %j = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index

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

          // Multi-tile global load A: mm_pos=mm, nn_pos=kk, cond_iter=nn
          // 2-level descriptor: m_pos/n_pos=base positions, mm_pos/nn_pos=tile indices
          %elt_size_a = arith.constant 2 : index
          %global_stride_a = affine.apply affine_map<()[SIZE_K, elt_sz] -> (SIZE_K * elt_sz)>()[%SIZE_K, %elt_size_a]
          %tensor_desc_a = aster_utils.struct_create(%a_global, %m_pos, %k_pos, %global_stride_a, %mm, %kk, %elt_size_a) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %cond_desc_a = aster_utils.struct_create(%k, %nn, %NT_M, %NT_K) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          func.call @maybe_global_load_multi_tile_coalesced(
            %cond_desc_a, %tensor_desc_a, %a_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>) -> ()

          // Multi-tile global load B: mm_pos=nn, nn_pos=kk, cond_iter=mm
          %elt_size_b = arith.constant 2 : index
          %global_stride_b = affine.apply affine_map<()[SIZE_K, elt_sz] -> (SIZE_K * elt_sz)>()[%SIZE_K, %elt_size_b]
          %tensor_desc_b = aster_utils.struct_create(%b_global, %n_pos, %k_pos, %global_stride_b, %nn, %kk, %elt_size_b) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %cond_desc_b = aster_utils.struct_create(%k, %mm, %NT_N, %NT_K) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          func.call @maybe_global_load_multi_tile_coalesced(
            %cond_desc_b, %tensor_desc_b, %b_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>) -> ()

          // Multi-tile DS writes
          // LDS stride in bytes = TILE_SIZE_K * 2 (f16 element size)
          %elt_size_lds = arith.constant 2 : index
          %lds_stride_bytes = affine.apply affine_map<()[TILE_SIZE_K, elt_sz] -> (TILE_SIZE_K * elt_sz)>()[%TILE_SIZE_K, %elt_size_lds]

          // Multi-tile DS write A: mm_pos=mm, nn_pos=kk (tile indices), cond_iter=nn
          %lds_desc_a = aster_utils.struct_create(%lds_a_base_off, %mm, %kk, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          %cond_desc_lds_a = aster_utils.struct_create(%k, %nn, %NT_M, %NT_K) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          func.call @maybe_lds_write_multi_tile_coalesced(
            %cond_desc_lds_a, %lds_desc_a, %a_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()

          // Multi-tile DS write B: mm_pos=nn, nn_pos=kk (tile indices), cond_iter=mm
          %lds_desc_b = aster_utils.struct_create(%lds_b_base_off, %nn, %kk, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          %cond_desc_lds_b = aster_utils.struct_create(%k, %mm, %NT_N, %NT_K) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          func.call @maybe_lds_write_multi_tile_coalesced(
            %cond_desc_lds_b, %lds_desc_b, %b_load_memref)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()

          // Create tile index and dimension descriptors
          %idx_desc = aster_utils.struct_create(%k, %mm, %nn, %kk) : (index, index, index, index) -> !tile_index_descriptor_4d
          %dims_desc = aster_utils.struct_create(%K, %MM, %NN, %KK) : (index, index, index, index) -> !tile_dims_descriptor_4d

          func.call @maybe_init_C(%idx_desc, %dims_desc, %c_fragments)
              {sched.delay = 0 : i64, sched.rate = 1 : i64}
            : (!tile_index_descriptor_4d, !tile_dims_descriptor_4d, memref<?x?x!vx4>) -> ()

          // LDS reads
          // LDS read A: ii=mm, jj=kk, cond_iter=nn
          %cond_desc_read_a = aster_utils.struct_create(%k, %nn, %c0, %c0) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          %lds_desc_read_a_base = aster_utils.struct_create(%lds_a_base_off, %c0, %c0, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          func.call @maybe_lds_read(
            %cond_desc_read_a, %lds_desc_read_a_base,
            %mm, %kk,
            %a_frag_memref)
              {sched.delay = 4 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // LDS read B: ii=nn, jj=kk, cond_iter=mm
          %cond_desc_read_b = aster_utils.struct_create(%k, %mm, %c0, %c0) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          %lds_desc_read_b_base = aster_utils.struct_create(%lds_b_base_off, %c0, %c0, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          func.call @maybe_lds_read(
            %cond_desc_read_b, %lds_desc_read_b_base,
            %nn, %kk,
            %b_frag_memref)
              {sched.delay = 4 : i64, sched.rate = 1 : i64}
            : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d,
              index, index,
              memref<?x?x?x!vx2>) -> ()

          // MFMA (decoupled from LDS reads via memrefs)
          func.call @maybe_mfma(%idx_desc, %a_frag_memref, %b_frag_memref, %c_fragments)
              {sched.delay = 10 : i64, sched.rate = 1 : i64}
            : (!tile_index_descriptor_4d, memref<?x?x?x!vx2>, memref<?x?x?x!vx2>, memref<?x?x!vx4>) -> ()

          // Store C fragment back to global memory
          %elt_size_c = arith.constant 4 : index // f32 size in bytes
          %GLOBAL_C_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_N, elt_sz] -> (SIZE_N * elt_sz)>()[%SIZE_N, %elt_size_c]
          %tensor_desc_c = aster_utils.struct_create(%c_global, %m_pos, %n_pos, %GLOBAL_C_STRIDE_IN_BYTES, %elt_size_c) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
          func.call @maybe_store_c_fragment(%idx_desc, %dims_desc, %c_fragments, %tensor_desc_c)
              {sched.delay = 12 : i64, sched.rate = 1 : i64}
          : (!tile_index_descriptor_4d, !tile_dims_descriptor_4d, memref<?x?x!vx4>, !tensor_position_descriptor_2d) -> ()

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
