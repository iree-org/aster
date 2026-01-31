// Drive this through pytest, only check input validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/64/g' -e 's/{{SIZE_N}}/64/g' -e 's/{{SIZE_K}}/64/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_M}}/4/g' -e 's/{{LOOP_SIZE_N}}/4/g' -e 's/{{LOOP_SIZE_K}}/4/g' \
// RUN: | sed -e 's/{{SIZE_K_BY_TILE_SIZE_K}}/2/g' \
// RUN: | sed -e 's/{{LOOP_SIZE_D_MMNNKK}}/6/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/descriptors.mlir,%p/library/common/copies.mlir,%p/library/common/multi-tile-copies.mlir" \
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

amdgcn.module @kernel_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // From indexing.mlir
  func.func private @tiled_grid_partition_2d(!index_pair, !index_pair) -> !index_pair
  // From register-init.mlir
  func.func private @init_vgprx4(i32) -> !vx4
  // From copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    !lds_position_descriptor_2d, i1) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
  // From multi-tile-copies.mlir
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
    !tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>)

  // Perform MFMA: load fragments, compute, store result fragment
  func.func private @mfma(
    %k: index, %mm: index, %nn: index, %kk: index,
    %a_frag_memref: memref<?x?x?x!vx2>,
    %b_frag_memref: memref<?x?x?x!vx2>,
    %c_fragments: memref<?x?x!vx4>
  ) {
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
    %c16 = arith.constant 16 : index
    %elt_sz_in_b = arith.constant 2 : index
    %false = arith.constant false

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

    // Allocate memrefs for decoupled global loads -> DS writes (linearized: [NT_I*NT_J])
    %NT_MK = affine.apply affine_map<()[NT_M, NT_K] -> (NT_M * NT_K)>()[%NT_M, %NT_K]
    %NT_NK = affine.apply affine_map<()[NT_N, NT_K] -> (NT_N * NT_K)>()[%NT_N, %NT_K]
    %a_load_memref = memref.alloca(%NT_MK) : memref<?x!vx2>
    %b_load_memref = memref.alloca(%NT_NK) : memref<?x!vx2>

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

          // Multi-tile global load A: execute when nn == 0 AND mm % NT_M == 0 AND kk % NT_K == 0
          %cond_nn_zero = arith.cmpi eq, %nn, %c0 : index
          %mm_mod_NT_M = affine.apply affine_map<()[mm, NT_M] -> (mm mod NT_M)>()[%mm, %NT_M]
          %kk_mod_NT_K = affine.apply affine_map<()[kk, NT_K] -> (kk mod NT_K)>()[%kk, %NT_K]
          %mm_aligned = arith.cmpi eq, %mm_mod_NT_M, %c0 : index
          %kk_aligned_a = arith.cmpi eq, %kk_mod_NT_K, %c0 : index
          %cond_load_a_1 = arith.andi %cond_nn_zero, %mm_aligned : i1
          %cond_load_a = arith.andi %cond_load_a_1, %kk_aligned_a : i1
          scf.if %cond_load_a {
            %elt_size_a = arith.constant 2 : index
            %global_stride_a = affine.apply affine_map<()[SIZE_K, elt_sz] -> (SIZE_K * elt_sz)>()[%SIZE_K, %elt_size_a]
            // mm_pos_load and kk_pos_load are the tile-level positions (in elements, i.e. *16)
            %mm_pos_load = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
            %kk_pos_load = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %tensor_desc_a = aster_utils.struct_create(%a_global, %m_pos, %k_pos, %global_stride_a, %mm_pos_load, %kk_pos_load, %elt_size_a) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
            func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
              %tensor_desc_a, %NT_M, %NT_K, %a_load_memref)
              : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
          }

          // Multi-tile global load B: execute when mm == 0 AND nn % NT_N == 0 AND kk % NT_K == 0
          %cond_mm_zero = arith.cmpi eq, %mm, %c0 : index
          %nn_mod_NT_N = affine.apply affine_map<()[nn, NT_N] -> (nn mod NT_N)>()[%nn, %NT_N]
          %kk_mod_NT_K_b = affine.apply affine_map<()[kk, NT_K] -> (kk mod NT_K)>()[%kk, %NT_K]
          %nn_aligned = arith.cmpi eq, %nn_mod_NT_N, %c0 : index
          %kk_aligned_b = arith.cmpi eq, %kk_mod_NT_K_b, %c0 : index
          %cond_load_b_1 = arith.andi %cond_mm_zero, %nn_aligned : i1
          %cond_load_b = arith.andi %cond_load_b_1, %kk_aligned_b : i1
          scf.if %cond_load_b {
            %elt_size_b = arith.constant 2 : index
            %global_stride_b = affine.apply affine_map<()[SIZE_K, elt_sz] -> (SIZE_K * elt_sz)>()[%SIZE_K, %elt_size_b]
            // nn_pos_load and kk_pos_load_b are the tile-level positions (in elements, i.e. *16)
            %nn_pos_load = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]
            %kk_pos_load_b = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %tensor_desc_b = aster_utils.struct_create(%b_global, %n_pos, %k_pos, %global_stride_b, %nn_pos_load, %kk_pos_load_b, %elt_size_b) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
            func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
              %tensor_desc_b, %NT_N, %NT_K, %b_load_memref)
              : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
          }

          // Multi-tile DS writes
          // LDS stride in bytes = TILE_SIZE_K * 2 (f16 element size)
          %elt_size_lds = arith.constant 2 : index
          %lds_stride_bytes = affine.apply affine_map<()[TILE_SIZE_K, elt_sz] -> (TILE_SIZE_K * elt_sz)>()[%TILE_SIZE_K, %elt_size_lds]

          // Multi-tile DS write A: execute when nn == 0 AND mm/kk aligned
          scf.if %cond_load_a {
            // mm_pos_lds and kk_pos_lds are the tile-level positions (in elements, i.e. *16)
            %mm_pos_lds = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
            %kk_pos_lds = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %lds_desc_a = aster_utils.struct_create(%lds_a_base_off, %mm_pos_lds, %kk_pos_lds, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
            func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
              %lds_desc_a, %NT_M, %NT_K, %a_load_memref)
              : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
          }

          // Multi-tile DS write B: execute when mm == 0 AND nn/kk aligned
          scf.if %cond_load_b {
            // nn_pos_lds and kk_pos_lds_b are the tile-level positions (in elements, i.e. *16)
            %nn_pos_lds = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]
            %kk_pos_lds_b = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %lds_desc_b = aster_utils.struct_create(%lds_b_base_off, %nn_pos_lds, %kk_pos_lds_b, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
            func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
              %lds_desc_b, %NT_N, %NT_K, %b_load_memref)
              : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()
          }

          // Initialize C fragment (only at first K iteration: k == 0 && kk == 0)
          %k_is_zero = arith.cmpi eq, %k, %c0 : index
          %kk_is_zero = arith.cmpi eq, %kk, %c0 : index
          %cond_init_c = arith.andi %k_is_zero, %kk_is_zero : i1
          scf.if %cond_init_c {
            %zero_i32 = arith.constant 0 : i32
            %zero_acc = func.call @init_vgprx4(%zero_i32) : (i32) -> !vx4
            memref.store %zero_acc, %c_fragments[%mm, %nn] : memref<?x?x!vx4>
          }

          // LDS read A: execute when nn == 0 (tile reuse across nn iterations)
          scf.if %cond_nn_zero {
            %mm_pos_a = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
            %kk_pos_a = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %lds_desc_read_a = aster_utils.struct_create(%lds_a_base_off, %mm_pos_a, %kk_pos_a, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
            %a_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(
              %lds_desc_read_a, %false)

              : (!lds_position_descriptor_2d, i1) -> !vx2
            memref.store %a_frag, %a_frag_memref[%k, %mm, %kk] : memref<?x?x?x!vx2>
          }

          // LDS read B: execute when mm == 0 (tile reuse across mm iterations)
          scf.if %cond_mm_zero {
            %nn_pos_b = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]
            %kk_pos_b = affine.apply affine_map<()[kk] -> (kk * 16)>()[%kk]
            %lds_desc_read_b = aster_utils.struct_create(%lds_b_base_off, %nn_pos_b, %kk_pos_b, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
            %b_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(
              %lds_desc_read_b, %false)

              : (!lds_position_descriptor_2d, i1) -> !vx2
            memref.store %b_frag, %b_frag_memref[%k, %nn, %kk] : memref<?x?x?x!vx2>
          }

          // MFMA (decoupled from LDS reads via memrefs)
          func.call @mfma(%k, %mm, %nn, %kk, %a_frag_memref, %b_frag_memref, %c_fragments)

            : (index, index, index, index, memref<?x?x?x!vx2>, memref<?x?x?x!vx2>, memref<?x?x!vx4>) -> ()

          // Store C fragment back to global memory (only at last K iteration)
          %K_minus_1 = affine.apply affine_map<()[K] -> (K - 1)>()[%K]
          %KK_minus_1 = affine.apply affine_map<()[KK] -> (KK - 1)>()[%KK]
          %k_is_last = arith.cmpi eq, %k, %K_minus_1 : index
          %kk_is_last = arith.cmpi eq, %kk, %KK_minus_1 : index
          %cond_store_c = arith.andi %k_is_last, %kk_is_last : i1
          scf.if %cond_store_c {
            %elt_size_c = arith.constant 4 : index // f32 size in bytes
            %GLOBAL_C_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_N, elt_sz] -> (SIZE_N * elt_sz)>()[%SIZE_N, %elt_size_c]
            %mm_pos_c = affine.apply affine_map<()[mm] -> (mm * 16)>()[%mm]
            %nn_pos_c = affine.apply affine_map<()[nn] -> (nn * 16)>()[%nn]
            %tensor_desc_c = aster_utils.struct_create(%c_global, %m_pos, %n_pos, %GLOBAL_C_STRIDE_IN_BYTES, %mm_pos_c, %nn_pos_c, %elt_size_c) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
            %acc = memref.load %c_fragments[%mm, %nn] : memref<?x?x!vx4>
            // Use transposed=true because MFMA C output has 4 elements in consecutive rows,
            // not consecutive columns. The "transposed" path does individual dword stores
            // with proper row offsets.
            %true_store = arith.constant true
            func.call @global_store_wave_16x16xf32_C_fragment_wait(%acc, %tensor_desc_c, %true_store)
              : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
          }

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
