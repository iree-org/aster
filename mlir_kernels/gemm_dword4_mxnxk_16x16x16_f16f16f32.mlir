// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | sed -e 's/{{SIZE_M}}/4/g' -e 's/{{SIZE_N}}/4/g' -e 's/{{SIZE_K}}/4/g' -e 's/{{LDS_B_SHIFT}}/8192/g' -e 's/{{LDS_SIZE}}/16384/g' \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

// CHECK-LABEL: amdgcn.module
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!index_pair = !aster_utils.struct<i: index, j: index>

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
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_grid_partition_2d(!index_pair, !index_pair) -> !index_pair
  // copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(
    !sx2, index, index, index, index, index, index, index) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    index, index, index, index, i1) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !sx2, index, index, index, index, index, i1) -> ()

  // Compute the wavefront-level contraction using MFMA instructions
  func.func private @wavefront_contract(
    %lds_a_base: index, %lds_b_base: index,         // LDS base offsets
    %TILE_SIZE_K: index,                            // The tile size in the reduction dimension
    %ii_pos: index, %jj_pos: index, %kk_pos: index, // The mma tile positions
    %acc: !vx4                                      // The accumulator register range
  ) -> !vx4 {
    // Read A and B fragments, assuming B is transposed to simplify the problem and have a single version.
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[TILE_SIZE_K, elt_size] ->
      (TILE_SIZE_K * elt_size)>()[%TILE_SIZE_K, %elt_size]
    %false = arith.constant false
    %a_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(
        %lds_a_base, %ii_pos, %kk_pos, %LDS_STRIDE_IN_BYTES, %false)
      : (index, index, index, index, i1) -> !vx2
    %b_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(
        %lds_b_base, %jj_pos, %kk_pos, %LDS_STRIDE_IN_BYTES, %false)
     : (index, index, index, index, i1) -> !vx2
    // Perform MFMA operation: C = A * B + C
    %result = amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x16_f16>
      %acc, %a_frag, %b_frag, %acc : !vx2, !vx2, !vx4 -> !vx4
    return %result : !vx4
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

    // Initialize C fragments
    %c0_i32 = arith.constant 0 : i32
    scf.for %c = %c0 to %d_MMNN step %c1 {
      %c_fragment = func.call @init_vgprx4(%c0_i32) : (i32) -> !vx4
      memref.store %c_fragment, %c_fragments[%c] : memref<?x!vx4>
    } {aster.constexpr}


    // Loop over the k_pos dimension
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_K, elt_size]
      -> (SIZE_K * elt_size)>()[%SIZE_K, %elt_size]
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[TILE_SIZE_K, elt_size]
      -> (TILE_SIZE_K * elt_size)>()[%TILE_SIZE_K, %elt_size]
    scf.for %k = %c0 to %K step %c1 {
      %k_pos = affine.apply affine_map<(tile_size)[idx] -> (idx * tile_size)>(%TILE_SIZE_K)[%k]

      // Load A tile into LDS
      scf.for %d_mmkk = %c0 to %d_MMKK step %c1 {
        // Calculate mma tile indices
        %iikk = affine.apply affine_map<()[d_idx, w, W] -> (d_idx * W + w)>()[%d_mmkk, %w, %W]
        %ii, %kk = affine.delinearize_index %iikk into (%MM, %KK) : index, index

        // Calculate positions
        %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
        %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]
        func.call @global_load_to_lds_wave_16x16_f16_wait(
            %a_global, %lds_a_base_off, %i_pos, %k_pos, %GLOBAL_STRIDE_IN_BYTES, %ii_pos, %kk_pos, %LDS_STRIDE_IN_BYTES)
          : (!sx2, index, index, index, index, index, index, index) -> ()
      } {aster.constexpr}

      // Load B tile into LDS
      scf.for %d_nnkk = %c0 to %d_NNKK step %c1 {
        // Calculate mma tile indices
        %jjkk = affine.apply affine_map<()[d_idx, w, W] -> (d_idx * W + w)>()[%d_nnkk, %w, %W]
        %jj, %kk = affine.delinearize_index %jjkk into (%NN, %KK) : index, index

        // Calculate positions
        %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
        %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]
        func.call @global_load_to_lds_wave_16x16_f16_wait(
            %b_global, %lds_b_base_off, %j_pos, %k_pos, %GLOBAL_STRIDE_IN_BYTES, %jj_pos, %kk_pos, %LDS_STRIDE_IN_BYTES)
          : (!sx2, index, index, index, index, index, index, index) -> ()
      } {aster.constexpr}

      // Synchronize to ensure data is in LDS
      amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0 immutable
      amdgcn.sopp.sopp <s_barrier>

      // Perform block-level contraction
      scf.for %d_mmnnkk = %c0 to %d_MMNNKK step %c1 {
        // Calculate mma tile indices
        %d_mmnn, %kk = affine.delinearize_index %d_mmnnkk into (%d_MMNN, %KK) : index, index
        %mmnn = affine.apply affine_map<()[d_idx, w, W] -> (d_idx * W + w)>()[%d_mmnn, %w, %W]
        %ii, %jj = affine.delinearize_index %mmnn into (%MM, %NN) : index, index

        // Compute positions
        %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
        %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]
        %kk_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%kk]

        // Compute the wavefront-level contraction and update the C fragment
        %acc = memref.load %c_fragments[%d_mmnn] : memref<?x!vx4>
        %updated_acc = func.call @wavefront_contract(%lds_a_base_off, %lds_b_base_off, %TILE_SIZE_K , %ii_pos, %jj_pos, %kk_pos, %acc)
          : (index, index, index, index, index, index, !vx4) -> !vx4
        memref.store %updated_acc, %c_fragments[%d_mmnn] : memref<?x!vx4>
      } {aster.constexpr}

      // Synchronize to ensure all threads are done using LDS
      amdgcn.sopp.sopp <s_barrier>
    } {aster.constexpr}

    // Store C fragments to global memory
    scf.for %d_mmnn = %c0 to %d_MMNN step %c1 {
      %mmnn = affine.apply affine_map<()[d_idx, w, W] -> (d_idx * W + w)>()[%d_mmnn, %w, %W]
      %ii, %jj = affine.delinearize_index %mmnn into (%MM, %NN) : index, index

      // Compute positions
      %ii_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%ii]
      %jj_pos = affine.apply affine_map<()[idx] -> (idx * 16)>()[%jj]

      // Store the fragment
      %fragment = memref.load %c_fragments[%d_mmnn] : memref<?x!vx4>
      %GLOBAL_C_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_N] ->
        (SIZE_N * 4)>()[%SIZE_N]
      %true = arith.constant true
      func.call @global_store_wave_16x16xf32_C_fragment_wait(
          %fragment, %c_global, %i_pos, %j_pos, %GLOBAL_C_STRIDE_IN_BYTES, %ii_pos, %jj_pos, %true)
        : (!vx4, !sx2, index, index, index, index, index, i1) -> ()
    } {aster.constexpr}

    return
  }

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
