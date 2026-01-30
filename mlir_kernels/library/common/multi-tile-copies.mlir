// Multi-tile copy functions for AMDGCN kernels.
//
// Provides multi-tile variants of the copy primitives in copies.mlir.
// Multi-tile operations process multiple 16x16 tiles at once for better memory
// coalescing and to enable overlapped execution patterns.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir,%p/library/common/simple-copies.mlir,%p/library/common/copies.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!future_lds_write = !amdgcn.write_token<shared>

amdgcn.library @multi_tile_copies isa = [#amdgcn.isa<cdna3>] {

  //===--------------------------------------------------------------------===//
  // External function declarations
  //===--------------------------------------------------------------------===//

  // From copies.mlir - _future variants for token-aware operations
  func.func private @global_load_wave_256xf16_via_dwordx2_future(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !future_global_read_any
  func.func private @lds_write_wave_256xf16_via_dwordx2_future(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write

  //===--------------------------------------------------------------------===//
  // Multi-tile global loads via dwordx2
  //   256xf16 tiles (16x16 elements) with coalesced memory access
  //   (future + wait variants)
  //===--------------------------------------------------------------------===//
  // These functions use a _future/_wait pattern:
  // - _future: Core implementation that returns tokens for explicit wait control
  // - _wait: Calls _future, then waits on all tokens

  // Multi-tile global load returning array of futures (value + token).
  // Loads m_tiles x n_tiles 16x16 tiles from global memory WITHOUT waiting.
  // Futures are stored in provided memref.
  //
  // Parameters:
  //   %tensor_desc: 2-level tensor position descriptor
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %future_memref: output memref[m_tiles * n_tiles] for !future_global_read_any (linearized)
  //
  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %future_memref: memref<?x!future_global_read_any>
  ) {
    %c0 = arith.constant 0 : index

    // Extract fields from descriptor
    %ptr = aster_utils.struct_extract %tensor_desc["ptr"] : !tensor_position_descriptor_2level_2d -> !sx2
    %m_pos_base = aster_utils.struct_extract %tensor_desc["m_pos"] : !tensor_position_descriptor_2level_2d -> index
    %n_pos_base = aster_utils.struct_extract %tensor_desc["n_pos"] : !tensor_position_descriptor_2level_2d -> index
    %global_stride_in_bytes = aster_utils.struct_extract %tensor_desc["global_stride_in_bytes"] : !tensor_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %tensor_desc["mm_pos"] : !tensor_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %tensor_desc["nn_pos"] : !tensor_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %tensor_desc["elt_size"] : !tensor_position_descriptor_2level_2d -> index

    // Tile size is always 16x16
    %row_size = arith.constant 16 : index
    %col_size = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // Iterate over tile indices directly (ensures bounds are correct)
    scf.for %i = %c0 to %m_tiles step %c1 {
      scf.for %j = %c0 to %n_tiles step %c1 {
        // Compute positions from tile indices
        %mm_pos = affine.apply affine_map<()[base, i, row_size] -> (base + i * row_size)>()[%mm_pos_base, %i, %row_size]
        %nn_pos = affine.apply affine_map<()[base, j, col_size] -> (base + j * col_size)>()[%nn_pos_base, %j, %col_size]

        // Load the tile and get future
        %pos_desc = aster_utils.struct_create(%ptr, %m_pos_base, %n_pos_base, %global_stride_in_bytes, %mm_pos, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        %transfer_size = arith.constant 8 : index
        %wave_size = arith.constant 64 : index
        %transfer_desc = aster_utils.struct_create(%row_size, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
        %future = func.call @global_load_wave_256xf16_via_dwordx2_future(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !future_global_read_any

        // Store future using linearized index
        %idx = affine.apply affine_map<()[i, j, n] -> (i * n + j)>()[%i, %j, %n_tiles]
        memref.store %future, %future_memref[%idx] : memref<?x!future_global_read_any>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Multi-tile global load with waiting.
  // Calls _future variant and waits on all tokens via amdgcn.wait.
  //
  // CHECK-LABEL: func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
    %tensor_desc: !tensor_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %result_memref: memref<?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Allocate temp memref for futures (linearized)
    %num_tiles = affine.apply affine_map<()[m, n] -> (m * n)>()[%m_tiles, %n_tiles]
    %future_memref = memref.alloca(%num_tiles) : memref<?x!future_global_read_any>

    // Call future variant to issue all loads
    func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
      %tensor_desc, %m_tiles, %n_tiles, %future_memref)
      : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!future_global_read_any>) -> ()

    // TODO: Wait on all tokens and use only amdgcn-convert-waits pass
    // For now we continue using s_waitcnt directly for correctness.
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Extract values from futures and store in result_memref (linearized)
    scf.for %idx = %c0 to %num_tiles step %c1 {
      %future = memref.load %future_memref[%idx] : memref<?x!future_global_read_any>
      %value_any, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
      // TODO: Wait on all tokens and use only amdgcn-convert-waits pass
      // amdgcn.wait deps %token : !amdgcn.read_token<flat>
      %value = aster_utils.from_any %value_any : !vx2
      memref.store %value, %result_memref[%idx] : memref<?x!vx2>
    } {aster.constexpr}

    return
  }

  //===--------------------------------------------------------------------===//
  // Multi-tile LDS writes via dwordx2
  //   256xf16 tiles (16x16 elements) with coalesced LDS access
  //   (future + wait variants)
  //===--------------------------------------------------------------------===//
  // Multi-tile LDS write returning array of write tokens.
  // Writes m_tiles x n_tiles 16x16 tiles to LDS WITHOUT waiting.
  // Write tokens are stored in provided memref.
  //
  // Parameters:
  //   %lds_desc: 2-level LDS position descriptor
  //   %m_tiles, %n_tiles: number of tiles in M and N directions
  //   %values_memref: input memref[m_tiles * n_tiles] with values to write (linearized)
  //   %token_memref: output memref[m_tiles * n_tiles] for write tokens (linearized)
  //
  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_future
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %values_memref: memref<?x!vx2>,
    %token_memref: memref<?x!amdgcn.write_token<shared>>
  ) {
    %c0 = arith.constant 0 : index

    // Extract fields from descriptor
    %lds_base_off = aster_utils.struct_extract %lds_desc["lds_base"] : !lds_position_descriptor_2level_2d -> index
    %mm_pos_base = aster_utils.struct_extract %lds_desc["mm_pos"] : !lds_position_descriptor_2level_2d -> index
    %nn_pos_base = aster_utils.struct_extract %lds_desc["nn_pos"] : !lds_position_descriptor_2level_2d -> index
    %lds_stride_in_bytes = aster_utils.struct_extract %lds_desc["lds_stride_in_bytes"] : !lds_position_descriptor_2level_2d -> index
    %elt_size = aster_utils.struct_extract %lds_desc["elt_size"] : !lds_position_descriptor_2level_2d -> index

    // Tile size is always 16x16
    %row_size = arith.constant 16 : index
    %col_size = arith.constant 16 : index
    %total_rows = affine.apply affine_map<()[m_tiles] -> (16 * m_tiles)>()[%m_tiles]
    %total_cols = affine.apply affine_map<()[n_tiles] -> (16 * n_tiles)>()[%n_tiles]

    scf.for %mt = %c0 to %total_rows step %row_size {
      scf.for %nt = %c0 to %total_cols step %col_size {
        // Compute tile indices
        %i = affine.apply affine_map<()[mt, row_size] -> (mt ceildiv row_size)>()[%mt, %row_size]
        %j = affine.apply affine_map<()[nt, col_size] -> (nt ceildiv col_size)>()[%nt, %col_size]

        // Compute linear index for both value and token storage
        %J = affine.apply affine_map<()[total_cols, col_size] -> (total_cols ceildiv col_size)>()[%total_cols, %col_size]
        %idx = affine.apply affine_map<()[i, j, J] -> (i * J + j)>()[%i, %j, %J]

        // Load value from memref using linearized index
        %value = memref.load %values_memref[%idx] : memref<?x!vx2>

        // Compute minor-tile positions
        %mm_pos = affine.apply affine_map<()[base, mt] -> (base + mt)>()[%mm_pos_base, %mt]
        %nn_pos = affine.apply affine_map<()[base, nt] -> (base + nt)>()[%nn_pos_base, %nt]

        // Write the tile and get future
        %lds_pos_desc = aster_utils.struct_create(%lds_base_off, %mm_pos, %nn_pos, %lds_stride_in_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_size_lds = arith.constant 8 : index
        %wave_size_lds = arith.constant 64 : index
        %transfer_desc_lds = aster_utils.struct_create(%row_size, %transfer_size_lds, %wave_size_lds) : (index, index, index) -> !transfer_descriptor_2d
        %token = func.call @lds_write_wave_256xf16_via_dwordx2_future(%lds_pos_desc, %transfer_desc_lds, %value)
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write

        // Store token
        memref.store %token, %token_memref[%idx] : memref<?x!amdgcn.write_token<shared>>
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Multi-tile LDS write with waiting.
  // Calls _future variant and waits on all tokens via amdgcn.wait.
  //
  // CHECK-LABEL: func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    %lds_desc: !lds_position_descriptor_2level_2d,
    %m_tiles: index,
    %n_tiles: index,
    %values_memref: memref<?x!vx2>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Allocate temp memref for tokens (linearized)
    %num_tiles = affine.apply affine_map<()[m, n] -> (m * n)>()[%m_tiles, %n_tiles]
    %token_memref = memref.alloca(%num_tiles) : memref<?x!amdgcn.write_token<shared>>

    // Call future variant to issue all writes
    func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_future(
      %lds_desc, %m_tiles, %n_tiles, %values_memref, %token_memref)
      : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>, memref<?x!amdgcn.write_token<shared>>) -> ()

    // TODO: Wait on all tokens and use only amdgcn-convert-waits pass
    // For now we continue using s_waitcnt directly for correctness.
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // // Wait on all tokens
    // scf.for %idx = %c0 to %num_tiles step %c1 {
    //   %token = memref.load %token_memref[%idx] : memref<?x!amdgcn.write_token<shared>>
    //   amdgcn.wait deps %token : !amdgcn.write_token<shared>
    // } {aster.constexpr}

    return
  }

}
