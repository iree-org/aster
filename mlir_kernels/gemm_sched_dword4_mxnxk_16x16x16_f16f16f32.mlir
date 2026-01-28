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

!index_pair = !aster_utils.struct<i: index, j: index>

// A 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

// A 2-level 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions of the outer tile (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - mm_pos, nn_pos: row and column positions of the inner tile (in elements)
//   - elt_size: element size in bytes
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - m_pos, n_pos: row and column positions (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2-level 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - mm_pos, nn_pos: row and column positions of the minor tile (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

///===----------------------------------------------------------------------===//
/// The descriptors below are not yet generally useful, they will need to be
/// revisited once we have more experience with multi-wave.
///===----------------------------------------------------------------------===//
// A scheduling phase descriptor containing:
//   - phase: current phase (0=load, 1=compute, 2=store)
//   - k: outer k tile index
//   - d_mmnnkk: distributed inner tile index
!sched_phase_descriptor = !aster_utils.struct<phase: index, k: index, d_mmnnkk: index>

// A distributed tile dimensions descriptor containing:
//   - MM, NN, KK: tile counts in M, N, K dimensions
//   - d_MMKK, d_NNKK, d_MMNN: distributed tile counts
//   - w, W: wave id and wave count
//   - K: total K tiles (for store phase)
!distributed_dims_descriptor = !aster_utils.struct<MM: index, NN: index, KK: index, d_MMKK: index, d_NNKK: index, d_MMNN: index, w: index, W: index, K: index>

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
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    !lds_position_descriptor_2d, i1) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    !vx4, !tensor_position_descriptor_2level_2d, i1) -> ()

  // Phase 0a: Global loads if phase 0 (decoupled from DS writes via memrefs)
  func.func private @maybe_global_load(
    %sched_desc: !sched_phase_descriptor,
    %dims_desc: !distributed_dims_descriptor,
    %tensor_desc_a: !tensor_position_descriptor_2d,
    %tensor_desc_b: !tensor_position_descriptor_2d,
    %k_pos: index,
    %a_load_memref: memref<?x?x!vx2>, %b_load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index

    // Extract scheduling info
    %phase = aster_utils.struct_extract %sched_desc["phase"] : !sched_phase_descriptor -> index
    %k = aster_utils.struct_extract %sched_desc["k"] : !sched_phase_descriptor -> index
    %d_mmnnkk = aster_utils.struct_extract %sched_desc["d_mmnnkk"] : !sched_phase_descriptor -> index

    // Extract dimensions
    %MM = aster_utils.struct_extract %dims_desc["MM"] : !distributed_dims_descriptor -> index
    %NN = aster_utils.struct_extract %dims_desc["NN"] : !distributed_dims_descriptor -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !distributed_dims_descriptor -> index
    %d_MMKK = aster_utils.struct_extract %dims_desc["d_MMKK"] : !distributed_dims_descriptor -> index
    %d_NNKK = aster_utils.struct_extract %dims_desc["d_NNKK"] : !distributed_dims_descriptor -> index
    %w = aster_utils.struct_extract %dims_desc["w"] : !distributed_dims_descriptor -> index
    %W = aster_utils.struct_extract %dims_desc["W"] : !distributed_dims_descriptor -> index

    // Extract tensor A info
    %a_global = aster_utils.struct_extract %tensor_desc_a["ptr"] : !tensor_position_descriptor_2d -> !sx2
    %i_pos = aster_utils.struct_extract %tensor_desc_a["m_pos"] : !tensor_position_descriptor_2d -> index
    %GLOBAL_STRIDE_IN_BYTES = aster_utils.struct_extract %tensor_desc_a["global_stride_in_bytes"] : !tensor_position_descriptor_2d -> index
    %elt_size = aster_utils.struct_extract %tensor_desc_a["elt_size"] : !tensor_position_descriptor_2d -> index

    // Extract tensor B info
    %b_global = aster_utils.struct_extract %tensor_desc_b["ptr"] : !tensor_position_descriptor_2d -> !sx2
    %j_pos = aster_utils.struct_extract %tensor_desc_b["m_pos"] : !tensor_position_descriptor_2d -> index

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
        %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
        %ii_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%iikk, %num_rows]
        %pos_desc_a = aster_utils.struct_create(%a_global, %i_pos, %k_pos, %GLOBAL_STRIDE_IN_BYTES, %ii_pos, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %transfer_size_a = arith.constant 8 : index // dwordx2
        %wave_size_a = arith.constant 64 : index
        %transfer_desc_a = aster_utils.struct_create(%num_rows, %transfer_size_a, %wave_size_a) : (index, index, index) -> !transfer_descriptor_2d
        %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc_a, %transfer_desc_a) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)

        memref.store %loaded, %a_load_memref[%k, %d_mmkk] : memref<?x?x!vx2>
      }

      // Global load B tile (decoupled: stores to memref)
      %ii, %d_nnkk = affine.delinearize_index %d_mmnnkk into (%MM, %d_NNKK) : index, index
      %is_mm_zero = arith.cmpi eq, %ii, %c0 : index
      scf.if %is_mm_zero {
        %jjkk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_nnkk, %w, %W]
        %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
        %jj_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%jjkk, %num_rows]
        %pos_desc_b = aster_utils.struct_create(%b_global, %j_pos, %k_pos, %GLOBAL_STRIDE_IN_BYTES, %jj_pos, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %transfer_size_b = arith.constant 8 : index // dwordx2
        %wave_size_b = arith.constant 64 : index
        %transfer_desc_b = aster_utils.struct_create(%num_rows, %transfer_size_b, %wave_size_b) : (index, index, index) -> !transfer_descriptor_2d
        %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc_b, %transfer_desc_b) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)

        memref.store %loaded, %b_load_memref[%k, %d_nnkk] : memref<?x?x!vx2>
      }
    }
    return
  }

  // Phase 0b: DS writes if phase 0 (decoupled from global loads via memrefs)
  func.func private @maybe_lds_write(
    %sched_desc: !sched_phase_descriptor,
    %dims_desc: !distributed_dims_descriptor,
    %lds_desc_a: !lds_position_descriptor_2d,
    %lds_desc_b: !lds_position_descriptor_2d,
    %a_load_memref: memref<?x?x!vx2>, %b_load_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index

    // Extract scheduling info
    %phase = aster_utils.struct_extract %sched_desc["phase"] : !sched_phase_descriptor -> index
    %k = aster_utils.struct_extract %sched_desc["k"] : !sched_phase_descriptor -> index
    %d_mmnnkk = aster_utils.struct_extract %sched_desc["d_mmnnkk"] : !sched_phase_descriptor -> index

    // Extract dimensions
    %MM = aster_utils.struct_extract %dims_desc["MM"] : !distributed_dims_descriptor -> index
    %NN = aster_utils.struct_extract %dims_desc["NN"] : !distributed_dims_descriptor -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !distributed_dims_descriptor -> index
    %d_MMKK = aster_utils.struct_extract %dims_desc["d_MMKK"] : !distributed_dims_descriptor -> index
    %d_NNKK = aster_utils.struct_extract %dims_desc["d_NNKK"] : !distributed_dims_descriptor -> index
    %w = aster_utils.struct_extract %dims_desc["w"] : !distributed_dims_descriptor -> index
    %W = aster_utils.struct_extract %dims_desc["W"] : !distributed_dims_descriptor -> index

    // Extract LDS A info
    %lds_a_base_off = aster_utils.struct_extract %lds_desc_a["lds_base"] : !lds_position_descriptor_2d -> index
    %LDS_STRIDE_IN_BYTES = aster_utils.struct_extract %lds_desc_a["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc_a["elt_size"] : !lds_position_descriptor_2d -> index

    // Extract LDS B info
    %lds_b_base_off = aster_utils.struct_extract %lds_desc_b["lds_base"] : !lds_position_descriptor_2d -> index

    %is_phase_0 = arith.cmpi eq, %phase, %c0 : index
    scf.if %is_phase_0 {
      // DS write A tile (decoupled: reads from memref)
      %_jj, %d_mmkk = affine.delinearize_index %d_mmnnkk into (%NN, %d_MMKK) : index, index
      %is_nn_zero = arith.cmpi eq, %_jj, %c0 : index
      scf.if %is_nn_zero {
        %iikk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_mmkk, %w, %W]
        %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
        %ii_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%iikk, %num_rows]
        %loaded = memref.load %a_load_memref[%k, %d_mmkk] : memref<?x?x!vx2>
        %lds_pos_desc_a = aster_utils.struct_create(%lds_a_base_off, %ii_pos, %c0, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_size_lds_a = arith.constant 8 : index // dwordx2
        %wave_size_lds_a = arith.constant 64 : index
        %transfer_desc_lds_a = aster_utils.struct_create(%num_rows, %transfer_size_lds_a, %wave_size_lds_a) : (index, index, index) -> !transfer_descriptor_2d
        func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc_a, %transfer_desc_lds_a, %loaded)
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
      }

      // DS write B tile (decoupled: reads from memref)
      %ii, %d_nnkk = affine.delinearize_index %d_mmnnkk into (%MM, %d_NNKK) : index, index
      %is_mm_zero = arith.cmpi eq, %ii, %c0 : index
      scf.if %is_mm_zero {
        %jjkk = affine.apply affine_map<()[d_idx, wv, Wv] -> (d_idx * Wv + wv)>()[%d_nnkk, %w, %W]
        %num_rows = affine.apply affine_map<()[KK] -> (16 ceildiv KK)>()[%KK]
        %jj_pos = affine.apply affine_map<()[idx, num_rows] -> (idx * num_rows)>()[%jjkk, %num_rows]
        %loaded = memref.load %b_load_memref[%k, %d_nnkk] : memref<?x?x!vx2>
        %lds_pos_desc_b = aster_utils.struct_create(%lds_b_base_off, %jj_pos, %c0, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_size_lds_b = arith.constant 8 : index // dwordx2
        %wave_size_lds_b = arith.constant 64 : index
        %transfer_desc_lds_b = aster_utils.struct_create(%num_rows, %transfer_size_lds_b, %wave_size_lds_b) : (index, index, index) -> !transfer_descriptor_2d
        func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc_b, %transfer_desc_lds_b, %loaded)
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
      }
    }
    return
  }

  // Phase 1a: LDS reads if phase 1 (decoupled from mfma via memrefs)
  func.func private @maybe_lds_read(
    %sched_desc: !sched_phase_descriptor,
    %dims_desc: !distributed_dims_descriptor,
    %lds_desc_a: !lds_position_descriptor_2d,
    %lds_desc_b: !lds_position_descriptor_2d,
    %a_frag_memref: memref<?x?x!vx2>, %b_frag_memref: memref<?x?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract scheduling info
    %phase = aster_utils.struct_extract %sched_desc["phase"] : !sched_phase_descriptor -> index
    %k = aster_utils.struct_extract %sched_desc["k"] : !sched_phase_descriptor -> index
    %d_mmnnkk = aster_utils.struct_extract %sched_desc["d_mmnnkk"] : !sched_phase_descriptor -> index

    // Extract dimensions
    %MM = aster_utils.struct_extract %dims_desc["MM"] : !distributed_dims_descriptor -> index
    %NN = aster_utils.struct_extract %dims_desc["NN"] : !distributed_dims_descriptor -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !distributed_dims_descriptor -> index
    %d_MMNN = aster_utils.struct_extract %dims_desc["d_MMNN"] : !distributed_dims_descriptor -> index
    %w = aster_utils.struct_extract %dims_desc["w"] : !distributed_dims_descriptor -> index
    %W = aster_utils.struct_extract %dims_desc["W"] : !distributed_dims_descriptor -> index

    // Extract LDS info
    %lds_a_base_off = aster_utils.struct_extract %lds_desc_a["lds_base"] : !lds_position_descriptor_2d -> index
    %LDS_STRIDE_IN_BYTES = aster_utils.struct_extract %lds_desc_a["lds_stride_in_bytes"] : !lds_position_descriptor_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc_a["elt_size"] : !lds_position_descriptor_2d -> index
    %lds_b_base_off = aster_utils.struct_extract %lds_desc_b["lds_base"] : !lds_position_descriptor_2d -> index

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
      %false = arith.constant false
      %lds_pos_desc_a_read = aster_utils.struct_create(%lds_a_base_off, %ii_pos, %kk_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
      %a_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc_a_read, %false)
        : (!lds_position_descriptor_2d, i1) -> !vx2
      %lds_pos_desc_b_read = aster_utils.struct_create(%lds_b_base_off, %jj_pos, %kk_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
      %b_frag = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%lds_pos_desc_b_read, %false)
        : (!lds_position_descriptor_2d, i1) -> !vx2
      memref.store %a_frag, %a_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>
      memref.store %b_frag, %b_frag_memref[%k, %d_mmnnkk] : memref<?x?x!vx2>
    }
    return
  }

  // Perform MFMA if phase 1: load fragments, compute, store result
  func.func private @maybe_mfma(
    %sched_desc: !sched_phase_descriptor,
    %dims_desc: !distributed_dims_descriptor,
    %a_frag_memref: memref<?x?x!vx2>, %b_frag_memref: memref<?x?x!vx2>,
    %c_fragments: memref<?x!vx4>
  ) {
    %c1 = arith.constant 1 : index

    // Extract scheduling info
    %phase = aster_utils.struct_extract %sched_desc["phase"] : !sched_phase_descriptor -> index
    %k = aster_utils.struct_extract %sched_desc["k"] : !sched_phase_descriptor -> index
    %d_mmnnkk = aster_utils.struct_extract %sched_desc["d_mmnnkk"] : !sched_phase_descriptor -> index

    // Extract dimensions
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !distributed_dims_descriptor -> index
    %d_MMNN = aster_utils.struct_extract %dims_desc["d_MMNN"] : !distributed_dims_descriptor -> index

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
    %sched_desc: !sched_phase_descriptor,
    %dims_desc: !distributed_dims_descriptor,
    %tensor_desc_c: !tensor_position_descriptor_2d,
    %c_fragments: memref<?x!vx4>
  ) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Extract scheduling info
    %phase = aster_utils.struct_extract %sched_desc["phase"] : !sched_phase_descriptor -> index
    %k = aster_utils.struct_extract %sched_desc["k"] : !sched_phase_descriptor -> index
    %d_mmnnkk = aster_utils.struct_extract %sched_desc["d_mmnnkk"] : !sched_phase_descriptor -> index

    // Extract dimensions
    %MM = aster_utils.struct_extract %dims_desc["MM"] : !distributed_dims_descriptor -> index
    %NN = aster_utils.struct_extract %dims_desc["NN"] : !distributed_dims_descriptor -> index
    %KK = aster_utils.struct_extract %dims_desc["KK"] : !distributed_dims_descriptor -> index
    %d_MMNN = aster_utils.struct_extract %dims_desc["d_MMNN"] : !distributed_dims_descriptor -> index
    %w = aster_utils.struct_extract %dims_desc["w"] : !distributed_dims_descriptor -> index
    %W = aster_utils.struct_extract %dims_desc["W"] : !distributed_dims_descriptor -> index
    %K = aster_utils.struct_extract %dims_desc["K"] : !distributed_dims_descriptor -> index

    // Extract tensor C info
    %c_global = aster_utils.struct_extract %tensor_desc_c["ptr"] : !tensor_position_descriptor_2d -> !sx2
    %i_pos = aster_utils.struct_extract %tensor_desc_c["m_pos"] : !tensor_position_descriptor_2d -> index
    %j_pos = aster_utils.struct_extract %tensor_desc_c["n_pos"] : !tensor_position_descriptor_2d -> index
    %GLOBAL_STRIDE_IN_BYTES = aster_utils.struct_extract %tensor_desc_c["global_stride_in_bytes"] : !tensor_position_descriptor_2d -> index
    %elt_size_c = aster_utils.struct_extract %tensor_desc_c["elt_size"] : !tensor_position_descriptor_2d -> index

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
        %pos_desc_c = aster_utils.struct_create(%c_global, %i_pos, %j_pos, %GLOBAL_STRIDE_IN_BYTES, %ii_pos, %jj_pos, %elt_size_c) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %true_store = arith.constant true
        func.call @global_store_wave_16x16xf32_C_fragment_wait(%fragment, %pos_desc_c, %true_store) : (!vx4, !tensor_position_descriptor_2level_2d, i1) -> ()
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
    } {aster.constexpr}


    // M, N are fully distributed to blocks.
    // Loop over remaining 4-D tile **distributed** tile index (K, d_MMNNKK) in 2 phases:
    //   - Phase 0 loads to shared
    //   - Phase 1 computes
    //   - Phase 2 stores to global
    %num_phases = arith.constant 3 : index
    %ub = affine.apply affine_map<()[K, num_phases, d_MMNNKK] ->
         (K * num_phases * d_MMNNKK)>
      ()[%K, %num_phases, %d_MMNNKK]

    // Pre-compute strides (lifted outside the loop)
    %elt_size_ab = arith.constant 2 : index // f16 size in bytes
    %GLOBAL_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_K, elt_size] -> (SIZE_K * elt_size)>()[%SIZE_K, %elt_size_ab]
    %LDS_STRIDE_IN_BYTES = affine.apply affine_map<()[TILE_SIZE_K, elt_size] -> (TILE_SIZE_K * elt_size)>()[%TILE_SIZE_K, %elt_size_ab]
    %elt_size_c = arith.constant 4 : index // f32 size in bytes
    %GLOBAL_C_STRIDE_IN_BYTES = affine.apply affine_map<()[SIZE_N] -> (SIZE_N * 4)>()[%SIZE_N]

    // Create distributed dimensions descriptor (loop-invariant)
    %dims_desc = aster_utils.struct_create(%MM, %NN, %KK, %d_MMKK, %d_NNKK, %d_MMNN, %w, %W, %K) : (index, index, index, index, index, index, index, index, index) -> !distributed_dims_descriptor

    // Create tensor base descriptors (loop-invariant except k_pos)
    %tensor_desc_a = aster_utils.struct_create(%a_global, %i_pos, %c0, %GLOBAL_STRIDE_IN_BYTES, %elt_size_ab) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %tensor_desc_b = aster_utils.struct_create(%b_global, %j_pos, %c0, %GLOBAL_STRIDE_IN_BYTES, %elt_size_ab) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
    %tensor_desc_c = aster_utils.struct_create(%c_global, %i_pos, %j_pos, %GLOBAL_C_STRIDE_IN_BYTES, %elt_size_c) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d

    // Create LDS base descriptors (loop-invariant, m_pos/n_pos unused - set to 0)
    %lds_desc_a = aster_utils.struct_create(%lds_a_base_off, %c0, %c0, %LDS_STRIDE_IN_BYTES, %elt_size_ab) : (index, index, index, index, index) -> !lds_position_descriptor_2d
    %lds_desc_b = aster_utils.struct_create(%lds_b_base_off, %c0, %c0, %LDS_STRIDE_IN_BYTES, %elt_size_ab) : (index, index, index, index, index) -> !lds_position_descriptor_2d

    scf.for %idx = %c0 to %ub step %c1 {
      // Decompose linear index into 3D index
      %k, %phase, %d_mmnnkk = affine.delinearize_index %idx into (%K, %num_phases, %d_MMNNKK) : index, index, index
      %k_pos = affine.apply affine_map<(tile_size)[tile] -> (tile * tile_size)>(%TILE_SIZE_K)[%k]

      // Create scheduling descriptor (loop-variant)
      %sched_desc = aster_utils.struct_create(%phase, %k, %d_mmnnkk) : (index, index, index) -> !sched_phase_descriptor

      // Phase 0a: Global loads (decoupled from DS writes via memrefs)
      func.call @maybe_global_load(
        %sched_desc, %dims_desc,
        %tensor_desc_a, %tensor_desc_b, %k_pos,
        %a_load_memref, %b_load_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (!sched_phase_descriptor, !distributed_dims_descriptor,
           !tensor_position_descriptor_2d, !tensor_position_descriptor_2d, index,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 0b: DS writes (decoupled from global loads via memrefs)
      func.call @maybe_lds_write(
        %sched_desc, %dims_desc,
        %lds_desc_a, %lds_desc_b,
        %a_load_memref, %b_load_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (!sched_phase_descriptor, !distributed_dims_descriptor,
           !lds_position_descriptor_2d, !lds_position_descriptor_2d,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 1a: LDS reads (decoupled from mfma via memrefs)
      func.call @maybe_lds_read(
        %sched_desc, %dims_desc,
        %lds_desc_a, %lds_desc_b,
        %a_frag_memref, %b_frag_memref)
        {sched.delay = 0 : i64, sched.rate = 1 : i64}
        : (!sched_phase_descriptor, !distributed_dims_descriptor,
           !lds_position_descriptor_2d, !lds_position_descriptor_2d,
           memref<?x?x!vx2>, memref<?x?x!vx2>) -> ()

      // Phase 1b: MFMA (decoupled from LDS reads via memrefs)
      func.call @maybe_mfma(
        %sched_desc, %dims_desc,
        %a_frag_memref, %b_frag_memref, %c_fragments)
        {sched.delay = 6 : i64, sched.rate = 1 : i64}
        : (!sched_phase_descriptor, !distributed_dims_descriptor,
           memref<?x?x!vx2>, memref<?x?x!vx2>, memref<?x!vx4>) -> ()

      // Phase 2: Store C fragment back to global memory
      func.call @maybe_store_c_fragment(
        %sched_desc, %dims_desc, %tensor_desc_c, %c_fragments)
          {sched.delay = 12 : i64, sched.rate = 1 : i64}
        : (!sched_phase_descriptor, !distributed_dims_descriptor,
           !tensor_position_descriptor_2d, memref<?x!vx4>) -> ()

    } {aster.constexpr, sched.dims = array<i64: {{SIZE_K_BY_TILE_SIZE_K}}, 3, {{LOOP_SIZE_D_MMNNKK}}> }

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
