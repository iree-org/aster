// Common copy functions for AMDGCN kernels.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register_init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!a   = !amdgcn.agpr
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

amdgcn.library @common_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register_init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @swizzle_A_16x16xf16() -> (index, index)
  func.func private @swizzle_C_16x16xf32() -> (index, index)

  //===--------------------------------------------------------------------===//
  // Simple wave-level 16x16xf16 tile reads/writes
  //===--------------------------------------------------------------------===//
  // Read a 16x16xf16 tile from global memory to VGPRs, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the global_load).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @global_load_wave_16x16xf16_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most tile position
    %n_pos: index,                  // The inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // The inner-most stride **in bytes** in global memory
  ) -> !vx2 {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %mm_pos, %nn = func.call @lane_delinearize_2d(%num_rows, %num_cols) : (index, index) -> (index, index)
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %off_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the global load
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_global = amdgcn.flat.global_load #amdgcn.inst<global_load_dwordx2> %dst, %ptr[%off_reg]
      : !vx2, !sx2[!v] -> !vx2

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return %from_global : !vx2
  }

  // Write a 16x16xf16 tile from VGPRs to global memory, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the global_store).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @global_store_wave_16x16xf16_wait(
    %value: !vx2,                   // The value to write to global memory
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most tile position
    %n_pos: index,                  // The inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // The inner-most stride **in bytes** in global memory
  ) {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %mm_pos, %nn = func.call @lane_delinearize_2d(%num_rows, %num_cols) : (index, index) -> (index, index)
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %off_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the global store
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx2> %value, %ptr[%off_reg]
      : !vx2, !sx2[!v]

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // Read a 16x16xf16 tile from LDS to VGPRs, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the ds_read).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_read_wave_16x16xf16_wait(
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) -> !vx2 {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %mm_pos, %nn = func.call @lane_delinearize_2d(%num_rows, %num_cols) : (index, index) -> (index, index)
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %off_lds_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %dst, %off_lds_reg, offset = %lds_base_i32
      : !v, i32 -> !vx2

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  // Write a 16x16xf16 tile from VGPRs to LDS, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the ds_write).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_write_wave_16x16xf16_wait(
    %value: !vx2,               // The value to write to LDS
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %mm_pos, %nn = func.call @lane_delinearize_2d(%num_rows, %num_cols) : (index, index) -> (index, index)
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %off_lds_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the DS write
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %value, %off_lds_reg, offset = %lds_base_i32
      : !vx2, !v, i32

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return
  }

  // Simple variant: load a 16x16xf16 tile from global to LDS.
  func.func private @global_to_lds_wave_16x16xf16_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos_global: index,           // The global outer-most tile position
    %n_pos_global: index,           // The global inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // Global stride in bytes
    %lds_base: index,               // The local base offset in LDS
    %m_pos_lds: index,              // The LDS outer-most tile position
    %n_pos_lds: index,              // The LDS inner-most tile position
    %LDS_STRIDE_IN_BYTES: index     // LDS stride in bytes
  ) {
    %loaded = func.call @global_load_wave_16x16xf16_wait(
        %ptr, %m_pos_global, %n_pos_global, %GLOBAL_STRIDE_IN_BYTES)
      : (!sx2, index, index, index) -> !vx2
    func.call @lds_write_wave_16x16xf16_wait(
        %loaded, %lds_base, %m_pos_lds, %n_pos_lds, %LDS_STRIDE_IN_BYTES)
      : (!vx2, index, index, index, index) -> ()
    return
  }

  // Simple variant: load a 16x16xf16 tile from LDS to global.
  func.func private @lds_to_global_wave_16x16xf16_wait(
    %lds_base: index,               // The local base offset in LDS
    %m_pos_lds: index,              // The LDS outer-most tile position
    %n_pos_lds: index,              // The LDS inner-most tile position
    %LDS_STRIDE_IN_BYTES: index,    // LDS stride in bytes
    %ptr: !sx2,                     // The global base pointer
    %m_pos_global: index,           // The global outer-most tile position
    %n_pos_global: index,           // The global inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // Global stride in bytes
  ) {
    %loaded = func.call @lds_read_wave_16x16xf16_wait(
        %lds_base, %m_pos_lds, %n_pos_lds, %LDS_STRIDE_IN_BYTES)
      : (index, index, index, index) -> !vx2
    func.call @global_store_wave_16x16xf16_wait(
        %loaded, %ptr, %m_pos_global, %n_pos_global, %GLOBAL_STRIDE_IN_BYTES)
      : (!vx2, !sx2, index, index, index) -> ()
    return
  }

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
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most major-tile position
    %n_pos: index,                  // The inner-most major-tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // The inner-most size **in bytes**
    %mm_pos: index,                 // The outer-most minor-tile position
    %nn_pos: index,                 // The inner-most minor-tile position
    %num_rows: index                // The number of rows in the 256 elements
  ) -> !vx2 {
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

    // Calculate global offset
    %off_reg = func.call @tiledx2_matrix_offset(
      %m_pos, %n_pos, %mm_pos, %nn_pos, %mmm_pos, %nnn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index, index, index) -> !v

    // Perform the load
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %loaded = amdgcn.flat.global_load <global_load_dwordx2> %dst, %ptr[%off_reg]
      : !vx2, !sx2[!v] -> !vx2

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return %loaded : !vx2
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
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %value, %off_lds_reg, offset = %l_off_i32
      : !vx2, !v, i32

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
    amdgcn.flat.global_store <global_store_dword> %value, %ptr[%off_reg] : !v, !sx2[!v]
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // Swizzled fragment reads/writes
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
    // Compute the swizzled positions
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %mm_pos, %nn_pos = func.call @swizzle_A_16x16xf16() : () -> (index, index)
    %off_lds_reg = func.call @tiled_matrix_offset(
        %m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %dst, %off_lds_reg, offset = %lds_base_i32
      : !v, i32 -> !vx2

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  // Store the `C` fragment (16x16xf32) from VGPRs to global memory, in a
  // **synchronized fashion** (i.e. waitcnt 0 is inserted after each global_store).
  // The caller is responsible for embedding distribution information into the
  // positions. The callee computes and embeds the swizzled positions.
  // This function assumes a major/minor tile structure for the global positions.
  func.func private @global_store_wave_16x16xf32_swizzled_C_fragment_wait(
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

    // Compute the swizzled positions
    %mmm_pos, %nnn_pos = func.call @swizzle_C_16x16xf32() : () -> (index, index)

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
    } {amdgcn.constexpr}
    return
  }
}
