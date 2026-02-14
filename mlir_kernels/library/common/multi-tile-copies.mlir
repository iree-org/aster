// Multi-tile copy functions for AMDGCN kernels.
//
// Provides multi-tile variants of the copy primitives in copies.mlir.
// Multi-tile operations process multiple 16x16 tiles at once for better memory
// coalescing and to enable overlapped execution patterns.
//
//===-----------------------------------------------------------------------===//
// DESIGN NOTE: Linearized Return Value Descriptors for API Composition
//===-----------------------------------------------------------------------===//
//
// All return values (futures, tokens, values) use 1D linearized memrefs with
// offset descriptors (e.g., !return_value_descriptor_1d_vx2). This design is
// useful for API composition:
//
// 1. SROA and mem2reg compatibility: Return values stored in memrefs will be
//    subject to Scalar Replacement of Aggregates (SROA) and mem2reg passes.
//    These passes work best with simple 1D layouts where each element has a
//    unique, statically-determinable index.
//
// 2. Composable offsets: When composing operations (e.g., K-loop iterations),
//    each call can write to a different region of the same memref by passing
//    different offsets. This avoids needing separate allocations per iteration
//    and prevents value clobbering.
//
// 3. Descriptor uniformity: While position descriptors for memory operations
//    (tensor, LDS) may naturally be 2D or multi-level, return value descriptors
//    should always be linearized. The caller linearizes tile indices (e.g.,
//    i * n_tiles + j) and passes an offset to partition the output space.
//
// Example composition pattern:
//   %result_memref = memref.alloca(%K_times_num_tiles) : memref<?x!vx2>
//   scf.for %k = 0 to %K {
//     %offset = affine.apply (k * num_tiles)
//     %desc = struct_create(%result_memref, %offset) -> !return_value_descriptor_1d_vx2
//     call @multi_tile_load(..., %desc)  // writes to [offset, offset+num_tiles)
//   }

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/simple-copies.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!return_value_descriptor_1d_vx2 = !aster_utils.struct<memref: memref<?x!vx2>, offset: index>
!future_global_read_descriptor_1d = !aster_utils.struct<memref: memref<?x!future_global_read_any>, offset: index>
!future_lds_read_descriptor_1d = !aster_utils.struct<memref: memref<?x!future_lds_read_any>, offset: index>
!write_token_descriptor_1d = !aster_utils.struct<memref: memref<?x!future_lds_write>, offset: index>

//===-----------------------------------------------------------------------===//
// Wave-level, multi-tile global load instructions, parameterizable by
// !tensor_position_descriptor_2level_2d and tile counts (m_tiles, n_tiles).
//
// Loads m_tiles x n_tiles 16x16xf16 tiles (256xf16 each) via dwordx2.
//
// Return value design: Results stored in linearized 1D memref via descriptor.
// The 2D tile indices (i, j) are linearized to (i * n_tiles + j) + offset.
// This enables SROA/mem2reg optimization and composable K-loop patterns where
// each iteration writes to a distinct region of the same output memref.
//===-----------------------------------------------------------------------===//
amdgcn.library @multi_tile_global_load_to_vgpr_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From copies.mlir
  func.func private @global_load_wave_256xf16_via_dwordx2_future(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !future_global_read_any

  //===--------------------------------------------------------------------===//
  // Multi-tile global loads via dwordx2
  //   m_tiles x n_tiles 16x16xf16 tiles (256xf16 each)
  // (future + wait variants)
  //===--------------------------------------------------------------------===//
  // Loads multiple 16x16xf16 tiles from global memory to VGPRs cooperatively.
  // Each tile is 256 f16 elements loaded via dwordx2 (4xf16 per thread, 64 threads).
  //
  // Parameters:
  //   %tensor_desc: !tensor_position_descriptor_2level_2d (2D position for memory access)
  //     - ptr: base pointer
  //     - m_pos, n_pos: major tile position (element coordinates)
  //     - mm_pos, nn_pos: minor tile base position within major tile
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %future_desc / %result_desc: 1D linearized output descriptor
  //     - memref: output storage for m_tiles * n_tiles elements
  //     - offset: base index for this call (enables K-loop composition)
  //
  // The _future variant issues all loads without waiting, storing futures.
  // The _wait variant calls _future, waits via s_waitcnt, then extracts values.
  //
  // IMPORTANT: Return descriptors are always 1D linearized (not 2D) to enable
  // SROA/mem2reg and composable offset-based API patterns.

  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %future_desc: !future_global_read_descriptor_1d
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract memref and offset from future descriptor
    %future_memref = aster_utils.struct_extract %future_desc["memref"] : !future_global_read_descriptor_1d -> memref<?x!future_global_read_any>
    %future_offset = aster_utils.struct_extract %future_desc["offset"] : !future_global_read_descriptor_1d -> index

    // Extract fields from tensor descriptor
    %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
    %m_pos_base = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
    %n_pos_base = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
    %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

    // Tile size is always 16x16
    %n_rows = arith.constant 16 : index
    %n_cols = arith.constant 16 : index
    %transfer_size = arith.constant 8 : index
    %wave_size = arith.constant 64 : index

    // Iterate over tile indices
    scf.for %i = %c0 to %m_tiles step %c1 {
      scf.for %j = %c0 to %n_tiles step %c1 {
        // Compute positions from tile indices
        %mm_pos = affine.apply affine_map<()[base, i, n_rows] -> (base + i * n_rows)>()[%mm_pos_base, %i, %n_rows]
        %nn_pos = affine.apply affine_map<()[base, j, n_cols] -> (base + j * n_cols)>()[%nn_pos_base, %j, %n_cols]

        // Load the tile and get future
        %pos_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %mm_pos, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %transfer_desc = aster_utils.struct_create(%n_rows, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
        %future = func.call @global_load_wave_256xf16_via_dwordx2_future(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !future_global_read_any

        // Store future using linearized index with offset
        %idx = affine.apply affine_map<()[i, j, n] -> (i * n + j)>()[%i, %j, %n_tiles]
        %store_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %future_offset]
        memref.store %future, %future_memref[%store_idx] : memref<?x!future_global_read_any>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %result_desc: !return_value_descriptor_1d_vx2
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract memref and offset from result descriptor
    %result_memref = aster_utils.struct_extract %result_desc["memref"] : !return_value_descriptor_1d_vx2 -> memref<?x!vx2>
    %memref_offset = aster_utils.struct_extract %result_desc["offset"] : !return_value_descriptor_1d_vx2 -> index

    // Allocate temp memref for futures (linearized)
    %num_tiles = affine.apply affine_map<()[m, n] -> (m * n)>()[%m_tiles, %n_tiles]
    %future_memref = memref.alloca(%num_tiles) : memref<?x!future_global_read_any>

    // Create descriptor for future memref (offset=0 since locally allocated)
    %future_desc = aster_utils.struct_create(%future_memref, %c0) : (memref<?x!future_global_read_any>, index) -> !future_global_read_descriptor_1d

    // Call future variant to issue all loads
    func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
      %tensor_desc, %m_tiles, %n_tiles, %future_desc)
      : (!tensor_position_descriptor_2level_2d, index, index, !future_global_read_descriptor_1d) -> ()

    // Wait on all loads via s_waitcnt
    // TODO: use amdgcn-convert-waits pass instead
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Extract values from futures and store in result_memref (linearized, with offset)
    scf.for %idx = %c0 to %num_tiles step %c1 {
      %future = memref.load %future_memref[%idx] : memref<?x!future_global_read_any>
      %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    // TODO: use amdgcn-convert-waits pass instead
    // amdgcn.wait deps %token : !amdgcn.read_token<flat>
      %value = aster_utils.from_any %value_any : !vx2
      %store_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %memref_offset]
      memref.store %value, %result_memref[%store_idx] : memref<?x!vx2>
    } {aster.constexpr}

    return
  }
}

//===-----------------------------------------------------------------------===//
// Wave-level, multi-tile LDS read instructions for MFMA fragment layouts,
// parameterizable by !lds_position_descriptor_2level_2d, tile counts, and %transposed.
//
// Reads m_tiles x n_tiles 16x16xf16 tiles from LDS into MFMA "A" fragment layout.
//
// Return value design: Results stored in linearized 1D memref via descriptor.
// The 2D tile indices (i, j) are linearized to (i * n_tiles + j) + offset.
// This enables SROA/mem2reg optimization and composable K-loop patterns where
// each iteration writes to a distinct region of the same output memref.
//===-----------------------------------------------------------------------===//
amdgcn.library @multi_tile_lds_read_mfma_fragment_to_vgpr_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From copies.mlir
  func.func private @lds_read_A_wave_16x16xf16_fragment_future(!lds_position_descriptor_2d, i1) -> !future_lds_read_any

  //===--------------------------------------------------------------------===//
  // Multi-tile LDS reads for MFMA fragment A
  //   m_tiles x n_tiles 16x16xf16 tiles via ds_read_b64
  // (future + wait variants)
  //===--------------------------------------------------------------------===//
  // Reads multiple 16x16xf16 tiles from LDS into MFMA "A" fragment layout.
  // Each tile is 256 f16 elements read via ds_read_b64 (4xf16 per thread, 64 threads).
  //
  // Parameters:
  //   %lds_desc: !lds_position_descriptor_2level_2d (2D position for LDS access)
  //     - lds_base: base offset in LDS (bytes)
  //     - mm_pos, nn_pos: minor tile base position (element coordinates)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %transposed: swaps row/col indexing for transposed layout (B matrix as A^T)
  //   %future_desc / %result_desc: 1D linearized output descriptor
  //     - memref: output storage for m_tiles * n_tiles elements
  //     - offset: base index for this call (enables K-loop composition)
  //
  // Thread mapping follows MFMA 16x16 layout from @mfma_index_A_16x16xf16().
  //
  // The _future variant issues all reads without waiting, storing futures.
  // The _wait variant calls _future, waits via s_waitcnt, then extracts values.
  //
  // IMPORTANT: Return descriptors are always 1D linearized (not 2D) to enable
  // SROA/mem2reg and composable offset-based API patterns.

  // CHECK-LABEL: func.func private @lds_read_wave_multi_tile_16x16xf16_fragment_future
  func.func private @lds_read_wave_multi_tile_16x16xf16_fragment_future(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %transposed: i1,
    %future_desc: !future_lds_read_descriptor_1d
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract memref and offset from future descriptor
    %future_memref = aster_utils.struct_extract %future_desc["memref"] : !future_lds_read_descriptor_1d -> memref<?x!future_lds_read_any>
    %future_offset = aster_utils.struct_extract %future_desc["offset"] : !future_lds_read_descriptor_1d -> index

    // Extract fields from LDS descriptor
    %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %lds_desc["mm_pos"] : !lds_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %lds_desc["nn_pos"] : !lds_position_descriptor_2level_2d -> index
    %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2level_2d -> index

    // Tile size is always 16x16
    %n_rows = arith.constant 16 : index
    %n_cols = arith.constant 16 : index

    // Iterate over tile indices
    scf.for %i = %c0 to %m_tiles step %c1 {
      scf.for %j = %c0 to %n_tiles step %c1 {
        // Compute positions from tile indices
        %mm_pos = affine.apply affine_map<()[base, i, n_rows] -> (base + i * n_rows)>()[%mm_pos_base, %i, %n_rows]
        %nn_pos = affine.apply affine_map<()[base, j, n_cols] -> (base + j * n_cols)>()[%nn_pos_base, %j, %n_cols]

        // Create 1-level descriptor for the read primitive
        %lds_read_desc = aster_utils.struct_create(%lds_base_off, %mm_pos, %nn_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %future = func.call @lds_read_A_wave_16x16xf16_fragment_future(%lds_read_desc, %transposed)
          : (!lds_position_descriptor_2d, i1) -> !future_lds_read_any

        // Store future using linearized index with offset
        %idx = affine.apply affine_map<()[i, j, n] -> (i * n + j)>()[%i, %j, %n_tiles]
        %store_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %future_offset]
        memref.store %future, %future_memref[%store_idx] : memref<?x!future_lds_read_any>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // CHECK-LABEL: func.func private @lds_read_wave_multi_tile_16x16xf16_fragment_wait
  func.func private @lds_read_wave_multi_tile_16x16xf16_fragment_wait(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %transposed: i1,
    %result_desc: !return_value_descriptor_1d_vx2
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract memref and offset from result descriptor
    %result_memref = aster_utils.struct_extract %result_desc["memref"] : !return_value_descriptor_1d_vx2 -> memref<?x!vx2>
    %memref_offset = aster_utils.struct_extract %result_desc["offset"] : !return_value_descriptor_1d_vx2 -> index

    // Allocate temp memref for futures (linearized)
    %num_tiles = affine.apply affine_map<()[m, n] -> (m * n)>()[%m_tiles, %n_tiles]
    %future_memref = memref.alloca(%num_tiles) : memref<?x!future_lds_read_any>

    // Create descriptor for future memref (offset=0 since locally allocated)
    %future_desc = aster_utils.struct_create(%future_memref, %c0) : (memref<?x!future_lds_read_any>, index) -> !future_lds_read_descriptor_1d

    // Call future variant to issue all reads
    func.call @lds_read_wave_multi_tile_16x16xf16_fragment_future(
      %lds_desc, %m_tiles, %n_tiles, %transposed, %future_desc)
      : (!lds_position_descriptor_2level_2d, index, index, i1, !future_lds_read_descriptor_1d) -> ()

    // Wait on all reads via s_waitcnt
    // TODO: use amdgcn-convert-waits pass instead
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Extract values from futures and store in result_memref (linearized, with offset)
    scf.for %idx = %c0 to %num_tiles step %c1 {
      %future = memref.load %future_memref[%idx] : memref<?x!future_lds_read_any>
      %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
      // TODO: use amdgcn-convert-waits pass instead
      // amdgcn.wait deps %token : !amdgcn.read_token<shared>
      %value = aster_utils.from_any %value_any : !vx2
      %store_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %memref_offset]
      memref.store %value, %result_memref[%store_idx] : memref<?x!vx2>
    } {aster.constexpr}

    return
  }
}

//===-----------------------------------------------------------------------===//
// Wave-level, multi-tile LDS write instructions, parameterizable by
// !lds_position_descriptor_2level_2d and tile counts (m_tiles, n_tiles).
//
// Writes m_tiles x n_tiles 16x16xf16 tiles (256xf16 each) via ds_write_b64.
//
// Return value design: Input values and output tokens use linearized 1D memrefs
// via descriptors. The 2D tile indices (i, j) are linearized to
// (i * n_tiles + j) + offset. This enables SROA/mem2reg optimization and
// composable K-loop patterns where each iteration operates on a distinct
// region of the input/output memrefs.
//===-----------------------------------------------------------------------===//
amdgcn.library @multi_tile_lds_write_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From copies.mlir
  func.func private @lds_write_wave_256xf16_via_dwordx2_future(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write

  //===--------------------------------------------------------------------===//
  // Multi-tile LDS writes via ds_write_b64
  //   m_tiles x n_tiles 16x16xf16 tiles (256xf16 each)
  // (future + wait variants)
  //===--------------------------------------------------------------------===//
  // Writes multiple 16x16xf16 tiles from VGPRs to LDS cooperatively.
  // Each tile is 256 f16 elements written via ds_write_b64 (4xf16 per thread, 64 threads).
  //
  // Parameters:
  //   %lds_desc: !lds_position_descriptor_2level_2d (2D position for LDS access)
  //     - lds_base: base offset in LDS (bytes)
  //     - mm_pos, nn_pos: minor tile base position (element coordinates)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (2 for f16)
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %values_desc: 1D linearized input descriptor for values to write
  //     - memref: input storage with m_tiles * n_tiles elements
  //     - offset: base index for this call (enables K-loop composition)
  //   %token_desc: 1D linearized output descriptor for write tokens (_future only)
  //     - memref: output storage for m_tiles * n_tiles tokens
  //     - offset: base index for this call
  //
  // The _future variant issues all writes without waiting, storing tokens.
  // The _wait variant calls _future, then waits via s_waitcnt.
  //
  // IMPORTANT: Value and token descriptors are always 1D linearized (not 2D)
  // to enable SROA/mem2reg and composable offset-based API patterns.

  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_future
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %values_desc: !return_value_descriptor_1d_vx2,
    %token_desc: !write_token_descriptor_1d
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Extract memref and offset from values descriptor
    %values_memref = aster_utils.struct_extract %values_desc["memref"] : !return_value_descriptor_1d_vx2 -> memref<?x!vx2>
    %values_offset = aster_utils.struct_extract %values_desc["offset"] : !return_value_descriptor_1d_vx2 -> index

    // Extract memref and offset from token descriptor
    %token_memref = aster_utils.struct_extract %token_desc["memref"] : !write_token_descriptor_1d -> memref<?x!future_lds_write>
    %token_offset = aster_utils.struct_extract %token_desc["offset"] : !write_token_descriptor_1d -> index

    // Extract fields from LDS descriptor
    %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %lds_desc["mm_pos"] : !lds_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %lds_desc["nn_pos"] : !lds_position_descriptor_2level_2d -> index
    %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2level_2d -> index

    // Tile size is always 16x16
    %n_rows = arith.constant 16 : index
    %n_cols = arith.constant 16 : index

    // Iterate over tile indices
    scf.for %i = %c0 to %m_tiles step %c1 {
      scf.for %j = %c0 to %n_tiles step %c1 {
        // Compute linear index
        %idx = affine.apply affine_map<()[i, j, n] -> (i * n + j)>()[%i, %j, %n_tiles]

        // Load value from memref (with offset)
        %load_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %values_offset]
        %value = memref.load %values_memref[%load_idx] : memref<?x!vx2>

        // Compute minor-tile positions
        %mm_pos = affine.apply affine_map<()[base, i, n_rows] -> (base + i * n_rows)>()[%mm_pos_base, %i, %n_rows]
        %nn_pos = affine.apply affine_map<()[base, j, n_cols] -> (base + j * n_cols)>()[%nn_pos_base, %j, %n_cols]

        // Write the tile and get token
        %lds_pos_desc = aster_utils.struct_create(%lds_base_off, %mm_pos, %nn_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_size = arith.constant 8 : index
        %wave_size = arith.constant 64 : index
        %transfer_desc = aster_utils.struct_create(%n_rows, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
        %token = func.call @lds_write_wave_256xf16_via_dwordx2_future(%lds_pos_desc, %transfer_desc, %value)
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write

        // Store token (with offset)
        %store_idx = affine.apply affine_map<()[idx, offset] -> (idx + offset)>()[%idx, %token_offset]
        memref.store %token, %token_memref[%store_idx] : memref<?x!future_lds_write>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %values_desc: !return_value_descriptor_1d_vx2
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Allocate temp memref for tokens (linearized)
    %num_tiles = affine.apply affine_map<()[m, n] -> (m * n)>()[%m_tiles, %n_tiles]
    %token_memref = memref.alloca(%num_tiles) : memref<?x!future_lds_write>

    // Create descriptor for token memref (offset=0 since locally allocated)
    %token_desc = aster_utils.struct_create(%token_memref, %c0) : (memref<?x!future_lds_write>, index) -> !write_token_descriptor_1d

    // Call future variant to issue all writes
    func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(
      %lds_desc, %m_tiles, %n_tiles, %values_desc, %token_desc)
      : (!lds_position_descriptor_2level_2d, index, index, !return_value_descriptor_1d_vx2, !write_token_descriptor_1d) -> ()

    // Wait on all writes via s_waitcnt
    // TODO: use amdgcn-convert-waits pass instead
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return
  }
}
