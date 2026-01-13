// Common copy functions for AMDGCN kernels.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr_range<[? + 1]>
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx3 = !amdgcn.sgpr_range<[? + 3]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!a   = !amdgcn.agpr
!ax1 = !amdgcn.agpr_range<[? + 1]>
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax3 = !amdgcn.agpr_range<[? + 3]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

amdgcn.library @common_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register-init.mlir
  func.func private @alloc_vgpr() -> !v
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // indexing.mlir
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @mfma_index_16x16_helper() -> (index, index)
  func.func private @xor_swizzled_mfma_index_16xf16(index, index) -> (index, index)
  func.func private @mfma_index_A_16x16xf16() -> (index, index)
  func.func private @mfma_index_C_16x16xf32() -> (index, index)

  //===--------------------------------------------------------------------===//
  // Global <-> LDS
  //===--------------------------------------------------------------------===//
  // Loads from global memory to VGPRs, in a **synchronized fashion** (i.e.
  // waitcnt 0 are inserted after global_load).
  // This function cooperatively loads 256 f16 elements arranged in a 16x16 matrix
  // with a configurable number of rows with 64 global_load_dwordx2 operations
  // (1 per thread).
  //   - %num_rows =  1: tile is 1x64xdwordx2 (1x256xf16)
  //   - %num_rows =  2: tile is 2x32xdwordx2 (2x128xf16)
  //   - %num_rows =  4: tile is 4x16xdwordx2 ( 4x64xf16)
  //   - %num_rows =  8: tile is  8x8xdwordx2 ( 8x32xf16)
  //   - %num_rows = 16: tile is 16x4xdwordx2 (16x16xf16)
  //   - %num_rows = 32: tile is 32x2xdwordx2 ( 32x8xf16)
  //   - %num_rows = 64: tile is 64x1xdwordx2 ( 64x4xf16)
  // This can be configured for better global memory coalescing when %num_rows is not 1.
  // This is typically useful when %GLOBAL_STRIDE_IN_BYTES is 64xf16(or greater),
  // (resp. 32xf16, 16xf16, 8xf16, 4xf16, 2xf16, 1xf16).
  // We use an extra %num_rows (instead of just %GLOBAL_STRIDE_IN_BYTES) to give
  // the caller the option to use non-coalesced loads and obtain better flexibility
  // (e.g. useful for pipelining).
  //
  // Notes:
  // This models the read part of a more general copy function that can be
  // generalized to different number of elements and different transfer sizes.
  // The positions n_pos, m_pos, etc. are in number of elements; an adjustment
  // by transfer_size / elt_size is needed to get the global memory offset.
  //
  // TODO: if we communicated results via opaque ptr + mem2reg, we could make
  // this generic without templating MLIR.
  // TODO: also add a variant with upper bounds and buffer_load to handle boundary conditions.
  // TODO: add a static assert to enforce these.
   func.func private @global_load_wave_elt_2d_wait_impl(
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most major-tile position
    %n_pos: index,                  // The inner-most major-tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // The inner-most size **in bytes**
    %mm_pos: index,                 // The outer-most minor-tile position
    %nn_pos: index,                 // The inner-most minor-tile position
    %num_rows: index,               // The number of rows for the transfer, must divide %wave_size evenly
    %elt_size: index,               // The size of each element in bytes (used to map positions to addresses)
    %transfer_size: index,          // The size of each transfer in bytes
    %wave_size: index               // The number of elements per wave
  ) -> !aster_utils.any {

    %mod = affine.apply affine_map<()[wave_size, num_rows]
      -> (wave_size mod num_rows)>()[%wave_size, %num_rows]
    // static assert that %mod is 0
    scf.index_switch %mod
    case 0 {
      scf.yield
    }
    default {
      amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 42
    }

    %num_cols = affine.apply affine_map<()[wave_size, num_rows]
      -> (wave_size ceildiv num_rows)>()[%wave_size, %num_rows]

    // Get threadlocal positions within the minor tile
    %mmm_pos, %nnn = func.call @lane_delinearize_2d(%num_rows, %num_cols)
      : (index, index) -> (index, index)
    %nnn_pos = affine.apply affine_map<()[nnn, transfer_size, elt_size] ->
      (nnn * transfer_size ceildiv elt_size)>()[%nnn, %transfer_size, %elt_size]

    // Calculate global offset
    %off_reg = func.call @tiledx2_matrix_offset(
      %m_pos, %n_pos, %mm_pos, %nn_pos, %mmm_pos, %nnn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index, index, index) -> !v

    // Perform the load
    %res = scf.index_switch %transfer_size -> !aster_utils.any
    case 4 {
        %dst = func.call @alloc_vgprx1() : () -> (!vx1)
        %loaded, %tok_load = amdgcn.load global_load_dword dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx1) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx1
        scf.yield %any : !aster_utils.any
    }
    case 8 {
        %dst = func.call @alloc_vgprx2() : () -> (!vx2)
        %loaded, %tok_load = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx2) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx2
        scf.yield %any : !aster_utils.any
    }
    case 12 {
        %dst = func.call @alloc_vgprx3() : () -> (!vx3)
        %loaded, %tok_load = amdgcn.load global_load_dwordx3 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx3) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx3
        scf.yield %any : !aster_utils.any
    }
    case 16 {
        %dst = func.call @alloc_vgprx4() : () -> (!vx4)
        %loaded, %tok_load = amdgcn.load global_load_dwordx4 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx4
        scf.yield %any : !aster_utils.any
    }
    default {
        amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 43
        %c0 = arith.constant 0 : index
        %any = aster_utils.to_any %c0 : index
        scf.yield %any : !aster_utils.any
    }

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return %res : !aster_utils.any
  }


  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most major-tile position
    %n_pos: index,                  // The inner-most major-tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // The inner-most size **in bytes**
    %mm_pos: index,                 // The outer-most minor-tile position
    %nn_pos: index,                 // The inner-most minor-tile position
    %num_rows: index                // The number of rows in the 256 elements
  ) -> !vx2 {
    %elt_size = arith.constant 2 : index      // f16 size in bytes
    %transfer_size = arith.constant 8 : index // dwordx2 size in bytes
    %wave_size = arith.constant 64 : index    // 64 threads per wave

    %loaded = func.call @global_load_wave_elt_2d_wait_impl(
        %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %num_rows, %elt_size, %transfer_size, %wave_size)
      : (!sx2, index, index, index, index, index, index, index, index, index) -> (!aster_utils.any)

    %res = aster_utils.from_any %loaded : !vx2
    return %res : !vx2
  }

  // Writes %value to LDS, in a **synchronized fashion** (i.e. waitcnt 0 is
  // inserted after ds_write).
  // This function cooperatively writes 256 f16 elements arranged in a 16x16 matrix
  // with a configurable number of rows with 64 ds_write_b64 operations
  // (1 per thread).
  //   - %num_rows =  1: tile is 1x64xdwordx2 (1x256xf16)
  //   - %num_rows =  2: tile is 2x32xdwordx2 (2x128xf16)
  //   - %num_rows =  4: tile is 4x16xdwordx2 ( 4x64xf16)
  //   - %num_rows =  8: tile is  8x8xdwordx2 ( 8x32xf16)
  //   - %num_rows = 16: tile is 16x4xdwordx2 (16x16xf16)
  //   - %num_rows = 32: tile is 32x2xdwordx2 ( 32x8xf16)
  //   - %num_rows = 64: tile is 64x1xdwordx2 ( 64x4xf16)
  // This can be configured to match the memory coalescing needs of a producer
  // global_load_wave_256xf16_via_dwordx2_wait when %num_rows is not 1.
  // This is typically useful when %LDS_STRIDE_IN_BYTES is 64xf16 (or greater),
  // (resp. 32xf16, 16xf16, 8xf16, 4xf16, 2xf16, 1xf16).
  // We use an extra %num_rows (instead of just %LDS_STRIDE_IN_BYTES) to give
  // the caller the option to use non-coalesced writes and obtain better flexibility
  // (e.g. useful for pipelining).
  //
  // Notes:
  // This models the write part of a more general copy function that can be
  // generalized to different number of elements and different transfer sizes.
  // The positions nn_pos, mm_pos, etc. are in number of elements; an adjustment
  // by transfer_size / elt_size is needed to get the LDS offset.
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    %lds_base_off: index,        // The local base offset in LDS
    %mm_pos: index,              // The outer-most minor-tile position
    %nn_pos: index,              // The inner-most minor-tile position
    %LDS_STRIDE_IN_BYTES: index, // The inner-most size **in bytes**
    %num_rows: index,            // The number of rows in the 256 elements
    %value: !vx2                 // The value to write to LDS
  ) {
    // Constants that could become generic parameters if we communicated results
    // via opaque ptr + mem2reg.
    %elt_size = arith.constant 2 : index      // f16 size in bytes
    %transfer_size = arith.constant 8 : index // dwordx2 size in bytes
    %wave_size = arith.constant 64 : index    // 64 threads per wave

    %num_cols = affine.apply affine_map<()[wave_size, num_rows]
      -> (wave_size ceildiv num_rows)>()[%wave_size, %num_rows]

    // Get local positions within the minor tile
    %mmm_pos, %nnn = func.call @lane_delinearize_2d(%num_rows, %num_cols) : (index, index) -> (index, index)
    %nnn_pos = affine.apply affine_map<()[nnn, transfer_size, elt_size] ->
      (nnn * transfer_size ceildiv elt_size)>()[%nnn, %transfer_size, %elt_size]

    // Calculate offset into LDS
    %off_lds_reg = func.call @tiled_matrix_offset(
        %mm_pos, %nn_pos, %mmm_pos, %nnn_pos, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // DS write to LDS
    %l_off_i32 = arith.index_cast %lds_base_off : index to i32
    %tok_write = amdgcn.store ds_write_b64 data %value addr %off_lds_reg offset c(%l_off_i32) : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    // Wait for LDS write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return
  }

  // Load a 16x16xf16 tile from global memory to LDS within a single wave, in
  // a **synchronized fashion** (i.e. waitcnt 0 are inserted after global_load
  // and after lds_write).
  // This function cooperatively loads a 16x16xf16 tile from global memory to LDS
  // and forces num_rows to be 16, resulting in non-coalesced accesses in order
  // to preserve the 16x16 shape.
  // TODO: support different global_load and lds_write num_rows, to achieve a
  // reshape (only when we have a clear use for it).
  func.func private @global_load_to_lds_wave_16x16_f16_wait(
    %ptr: !sx2,                     // The global base pointer
    %lds_base_off: index,           // The local base offset in LDS
    %m_pos: index,                  // The outer-most major-tile position (in global memory)
    %n_pos: index,                  // The inner-most major-tile position (in global memory)
    %GLOBAL_STRIDE_IN_BYTES: index, // The inner-most size **in bytes** in global memory
    %mm_pos: index,                 // The outer-most minor-tile position (in LDS)
    %nn_pos: index,                 // The inner-most minor-tile position (in LDS)
    %LDS_STRIDE_IN_BYTES: index     // The inner-most major-tile size **in bytes** in LDS
  ) {
    %num_rows = arith.constant 16 : index
    %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(
        %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %num_rows)
      : (!sx2, index, index, index, index, index, index) -> (!vx2)
    func.call @lds_write_wave_256xf16_via_dwordx2_wait(
        %lds_base_off, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %num_rows, %loaded)
      : (index, index, index, index, index, !vx2) -> ()
    return
  }

  // Store a dword to global memory, in a **synchronized fashion** (i.e.
  // waitcnt 0 are inserted after global_store).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos (and make them workgroup/wave/thread/lane-dependent).
  func.func private @store_to_global_dword_wait(
    %value: !v,                     // Value to store
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most position
    %n_pos: index,                  // The inner-most position
    %GLOBAL_STRIDE_IN_BYTES: index  // The inner-most stride **in bytes** in global memory
  ) {
    %elt_size = arith.constant 4 : index // dword size in bytes
    %off_reg = func.call @matrix_offset(%m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index) -> !v
    %c0_store = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dword data %value addr %ptr offset d(%off_reg) + c(%c0_store) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // MFMA fragment reads/writes
  //===--------------------------------------------------------------------===//
  // Read the `A` fragment (16x16xf16) from LDS to VGPRs, in a **synchronized
  // fashion** (i.e. waitcnt 0 is inserted after the ds_read).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) -> !vx2 {
    // Compute the MFMA positions
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %mm_pos, %nn_pos = func.call @mfma_index_A_16x16xf16() : () -> (index, index)
    %off_lds_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds_reg offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  // Store the `C` fragment (16x16xf32) from VGPRs to global memory, in a
  // **synchronized fashion** (i.e. waitcnt 0 is inserted after each global_store).
  // The caller is responsible for embedding distribution information into the
  // positions. The callee computes and embeds the MFMA positions.
  // This function assumes a major/minor tile structure for the global positions.
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    %acc: !vx4,                     // The accumulator fragment to store
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most major-tile position
    %n_pos: index,                  // The inner-most major-tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // The inner-most stride **in bytes** in global memory
    %mm_pos: index,                 // The outer-most minor-tile position
    %nn_pos: index                  // The inner-most minor-tile position
  ) {
    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Split the fragment into 4 dword values
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %acc : !vx4
    %C_fragment =  memref.alloca() : memref<4x!v>
    memref.store %v0, %C_fragment[%c0] : memref<4x!v>
    memref.store %v1, %C_fragment[%c1] : memref<4x!v>
    memref.store %v2, %C_fragment[%c2] : memref<4x!v>
    memref.store %v3, %C_fragment[%c3] : memref<4x!v>

    // Compute the MFMA positions
    %mmm_pos, %nnn_pos = func.call @mfma_index_C_16x16xf32() : () -> (index, index)

    // Calculate global j position
    %n_global_pos = affine.apply
      affine_map<()[n_pos, nn_pos, nnn_pos] -> (n_pos + nn_pos + nnn_pos)>
      ()[%n_pos, %nn_pos, %nnn_pos]

    // Store each fragment to global memory
    scf.for %mmmm_pos = %c0 to %c4 step %c1 {
      %fragment = memref.load %C_fragment[%mmmm_pos] : memref<4x!v>
      // Calculate global i position
      %m_global_pos = affine.apply
        affine_map<()[m_pos, mm_pos, mmm_pos, mmmm_pos] -> (m_pos + mm_pos + mmm_pos + mmmm_pos)>
        ()[%m_pos, %mm_pos, %mmm_pos, %mmmm_pos]

      // Store to global memory with wait
      func.call @store_to_global_dword_wait(%fragment, %ptr, %m_global_pos, %n_global_pos, %GLOBAL_STRIDE_IN_BYTES)
        : (!v, !sx2, index, index, index) -> ()
    } {aster.constexpr}
    return
  }

  //===--------------------------------------------------------------------===//
  // Swizzled MFMA fragment reads/writes
  //===--------------------------------------------------------------------===//
  // Read the `A` fragment (16x16xf16) from LDS to VGPRs, in a **synchronized
  // fashion** (i.e. waitcnt 0 is inserted after the ds_read).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) -> !vx2 {
    %elt_size = arith.constant 2 : index // f16 size in bytes

    // Apply A-matrix swizzle
    %row, %col = func.call @mfma_index_A_16x16xf16() : () -> (index, index)
    %swizzled_row, %swizzled_col = func.call @xor_swizzled_mfma_index_16xf16(%row, %col)
      : (index, index) -> (index, index)
    %off_lds = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %swizzled_row, %swizzled_col, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %result, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %result : !vx2
  }

}
