// Simple copy functions for AMDGCN kernels (mostly for testing purposes).
//
// Provides simplified copy primitives with fixed 16x16xf16 tile sizes and
// fixed num_rows=16 thread distribution. For production use, prefer the
// parameterizable variants in copies.mlir.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!a   = !amdgcn.agpr
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax4 = !amdgcn.agpr_range<[? + 4]>
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>
!index_descriptor_3level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, iii: index, jjj: index, stride: index, elt_size_b: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

//===-----------------------------------------------------------------------===//
// Wave-level, simple copy instructions with fixed 16x16xf16 tile size,
// parameterizable by !tensor_position_descriptor_2d or !lds_position_descriptor_2d.
//
// Fixed num_rows=16 thread distribution (16x4 grid, each thread handles 4xf16).
// All variants use synchronized waits (s_waitcnt after each operation).
//===-----------------------------------------------------------------------===//
amdgcn.library @simple_copies isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @matrix_offset(!index_descriptor_2d) -> !v
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair

  //===--------------------------------------------------------------------===//
  // Simple global load: 16x16xf16 tile via global_load_dwordx2
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Loads a 16x16xf16 tile from global memory to VGPRs cooperatively.
  // Each thread loads 4xf16 (8 bytes) via dwordx2, totaling 256 f16 elements
  // across 64 threads in a 16x4 grid (num_rows=16 fixed).
  //
  // Parameters from !tensor_position_descriptor_2d:
  //   - ptr: base pointer
  //   - m_pos, n_pos: tile position in elements (row, col)
  //   - global_stride_in_bytes: row stride
  //   - elt_size: element size in bytes (2 for f16)
  //
  // Synchronization: inserts s_waitcnt vmcnt=0 after load.
  func.func private @simple_global_load_wave_16x16xf16_wait(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !vx2 {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "elt_size"] : !tensor_position_descriptor_2d -> !sx2, index, index, index, index

    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the global load
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_global, %tok_load = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return %from_global : !vx2
  }

  //===--------------------------------------------------------------------===//
  // Simple global store: 16x16xf16 tile via global_store_dwordx2
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Stores a 16x16xf16 tile from VGPRs to global memory cooperatively.
  // Each thread stores 4xf16 (8 bytes) via dwordx2, totaling 256 f16 elements
  // across 64 threads in a 16x4 grid (num_rows=16 fixed).
  //
  // Parameters:
  //   %value: !vx2 - the value to store
  //   %pos_desc: !tensor_position_descriptor_2d (same fields as load)
  //
  // Synchronization: inserts s_waitcnt vmcnt=0 after store.
  func.func private @simple_global_store_wave_16x16xf16_wait(
    %value: !vx2,
    %pos_desc: !tensor_position_descriptor_2d
  ) {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "elt_size"] : !tensor_position_descriptor_2d -> !sx2, index, index, index, index

    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the global store
    %c0 = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dwordx2 data %value addr %ptr offset d(%off_reg) + c(%c0) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // Simple LDS read: 16x16xf16 tile via ds_read_b64
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Reads a 16x16xf16 tile from LDS to VGPRs cooperatively.
  // Each thread reads 4xf16 (8 bytes) via ds_read_b64, totaling 256 f16 elements
  // across 64 threads in a 16x4 grid (num_rows=16 fixed).
  //
  // Parameters from !lds_position_descriptor_2d:
  //   - lds_base: base offset in LDS (bytes)
  //   - m_pos, n_pos: tile position in elements (row, col)
  //   - lds_stride_in_bytes: row stride
  //   - elt_size: element size in bytes (2 for f16)
  //
  // Synchronization: inserts s_waitcnt lgkmcnt=0 after read.
  func.func private @simple_lds_read_wave_16x16xf16_wait(
    %pos_desc: !lds_position_descriptor_2d
  ) -> !vx2 {
    %lds_base, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index

    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds_reg offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  //===--------------------------------------------------------------------===//
  // Simple LDS write: 16x16xf16 tile via ds_write_b64
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Writes a 16x16xf16 tile from VGPRs to LDS cooperatively.
  // Each thread writes 4xf16 (8 bytes) via ds_write_b64, totaling 256 f16 elements
  // across 64 threads in a 16x4 grid (num_rows=16 fixed).
  //
  // Parameters:
  //   %value: !vx2 - the value to write
  //   %pos_desc: !lds_position_descriptor_2d (same fields as read)
  //
  // Synchronization: inserts s_waitcnt lgkmcnt=0 after write.
  func.func private @simple_lds_write_wave_16x16xf16_wait(
    %value: !vx2,
    %pos_desc: !lds_position_descriptor_2d
  ) {
    %lds_base, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index

    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the DS write
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %tok_write = amdgcn.store ds_write_b64 data %value addr %off_lds_reg offset c(%lds_base_i32) : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // Simple global→LDS copy: 16x16xf16 tile
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Copies a 16x16xf16 tile from global memory to LDS.
  // Composes @simple_global_load_wave_16x16xf16_wait and
  // @simple_lds_write_wave_16x16xf16_wait.
  //
  // Parameters:
  //   %global_pos_desc: !tensor_position_descriptor_2d (source)
  //   %lds_pos_desc: !lds_position_descriptor_2d (destination)
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(
    %global_pos_desc: !tensor_position_descriptor_2d,
    %lds_pos_desc: !lds_position_descriptor_2d
  ) {
    %loaded = func.call @simple_global_load_wave_16x16xf16_wait(%global_pos_desc)
      : (!tensor_position_descriptor_2d) -> !vx2
    func.call @simple_lds_write_wave_16x16xf16_wait(%loaded, %lds_pos_desc)
      : (!vx2, !lds_position_descriptor_2d) -> ()
    return
  }

  //===--------------------------------------------------------------------===//
  // Simple LDS→global copy: 16x16xf16 tile
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Copies a 16x16xf16 tile from LDS to global memory.
  // Composes @simple_lds_read_wave_16x16xf16_wait and
  // @simple_global_store_wave_16x16xf16_wait.
  //
  // Parameters:
  //   %lds_pos_desc: !lds_position_descriptor_2d (source)
  //   %global_pos_desc: !tensor_position_descriptor_2d (destination)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(
    %lds_pos_desc: !lds_position_descriptor_2d,
    %global_pos_desc: !tensor_position_descriptor_2d
  ) {
    %loaded = func.call @simple_lds_read_wave_16x16xf16_wait(%lds_pos_desc)
      : (!lds_position_descriptor_2d) -> !vx2
    func.call @simple_global_store_wave_16x16xf16_wait(%loaded, %global_pos_desc)
      : (!vx2, !tensor_position_descriptor_2d) -> ()
    return
  }
}
