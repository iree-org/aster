// Common copy functions for AMDGCN kernels.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

// From descriptors.mlir
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
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>
!index_descriptor_3level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, iii: index, jjj: index, stride: index, elt_size_b: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

// A future descriptor for async operations containing:
//   - value: the loaded value (type-erased via !aster_utils.any)
//   - token: the read token for synchronization
// This enables callers to wait explicitly via amdgcn.wait instead of s_waitcnt.
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// A future descriptor for async LDS write operations containing:
//   - token: the write token for synchronization via amdgcn.wait
!future_lds_write = !amdgcn.write_token<shared>

// A future descriptor for async LDS read operations containing:
//   - value: the loaded value (type-erased via !aster_utils.any)
//   - token: the read token for synchronization via amdgcn.wait
!future_lds_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

// A future descriptor for async global write operations containing:
//   - token: the write token for synchronization via amdgcn.wait
!future_global_write = !amdgcn.write_token<flat>

amdgcn.library @common_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // From register-init.mlir
  func.func private @alloc_vgpr() -> !v
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  func.func private @alloc_sgprx2() -> !sx2
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @matrix_offset(!index_descriptor_2d) -> !v
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair

  //===--------------------------------------------------------------------===//
  // Undef future helpers (for unreachable default cases)
  //===--------------------------------------------------------------------===//
  // These return undef futures for scf.index_switch default branches that trap.
  // The returned values are never used since the trap terminates execution.

  // Issue a dummy global_load from address 0 to get a valid read token.
  // This code must be unreachable (after s_trap) but needed for type correctness.
  // Note: if we ever need this for real, consider an amdgcn.undef_token.
  func.func private @trapping_undef_future_global_read() -> !future_global_read_any {
    /// TRAP
    amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 44

    %addr = func.call @alloc_vgprx2() : () -> !vx2
    %dst = func.call @alloc_vgprx1() : () -> !vx1
    %loaded, %token = amdgcn.load global_load_dword dest %dst addr %addr
      : dps(!vx1) ins(!vx2) -> !amdgcn.read_token<flat>
    %any = aster_utils.to_any %loaded : !vx1
    %future = aster_utils.struct_create(%any, %token) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
    return %future : !future_global_read_any
  }

  // Issue a dummy global_store to address 0 to get a valid write token.
  // This code must be unreachable (after s_trap) but needed for type correctness.
  // Note: if we ever need this for real, consider an amdgcn.undef_token.
  func.func private @trapping_undef_future_global_write() -> !future_global_write {
    /// TRAP
    amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 45

    %addr = func.call @alloc_vgprx2() : () -> !vx2
    %data = func.call @alloc_vgprx1() : () -> !vx1
    %token = amdgcn.store global_store_dword data %data addr %addr
      : ins(!vx1, !vx2) -> !amdgcn.write_token<flat>
    return %token : !future_global_write
  }

  //===--------------------------------------------------------------------===//
  // Global loads, single dword/dwordx2/dwordx3/dwordx4 (wait + future variants)
  //===--------------------------------------------------------------------===//
  // Load from global memory implementation returning a future.
  // Supports dword (4 bytes), dwordx2 (8 bytes), dwordx3 (12 bytes), and
  // dwordx4 (16 bytes) transfers.
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos (and make them workgroup/wave/thread/lane-dependent).
  // Returns a future containing the value and token for explicit wait control.
  func.func private @load_from_global_impl(
    %pos_desc: !tensor_position_descriptor_2d,
    %transfer_size: index           // Transfer size in bytes (4, 8, 12, or 16)
  ) -> !future_global_read_any {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "elt_size"] : !tensor_position_descriptor_2d -> !sx2, index, index, index, index
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %transfer_size) : (index, index, index, index) -> !index_descriptor_2d
    %off_reg = func.call @matrix_offset(%desc) : (!index_descriptor_2d) -> !v
    %c0_load = arith.constant 0 : i32

    %res = scf.index_switch %transfer_size -> !future_global_read_any
    case 4 {
      %dst = func.call @alloc_vgprx1() : () -> (!vx1)
      %loaded, %token = amdgcn.load global_load_dword dest %dst addr %ptr offset d(%off_reg) + c(%c0_load)
        : dps(!vx1) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
      %any = aster_utils.to_any %loaded : !vx1
      %future = aster_utils.struct_create(%any, %token) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
      scf.yield %future : !future_global_read_any
    }
    case 8 {
      %dst = func.call @alloc_vgprx2() : () -> (!vx2)
      %loaded, %token = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0_load)
        : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
      %any = aster_utils.to_any %loaded : !vx2
      %future = aster_utils.struct_create(%any, %token) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
      scf.yield %future : !future_global_read_any
    }
    case 12 {
      %dst = func.call @alloc_vgprx3() : () -> (!vx3)
      %loaded, %token = amdgcn.load global_load_dwordx3 dest %dst addr %ptr offset d(%off_reg) + c(%c0_load)
        : dps(!vx3) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
      %any = aster_utils.to_any %loaded : !vx3
      %future = aster_utils.struct_create(%any, %token) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
      scf.yield %future : !future_global_read_any
    }
    case 16 {
      %dst = func.call @alloc_vgprx4() : () -> (!vx4)
      %loaded, %token = amdgcn.load global_load_dwordx4 dest %dst addr %ptr offset d(%off_reg) + c(%c0_load)
        : dps(!vx4) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
      %any = aster_utils.to_any %loaded : !vx4
      %future = aster_utils.struct_create(%any, %token) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
      scf.yield %future : !future_global_read_any
    }
    default {
      // Note: this is an unexpected path needed for completeness, it will trap.
      %future = func.call @trapping_undef_future_global_read() : () -> !future_global_read_any
      scf.yield %future : !future_global_read_any
    }

    return %res : !future_global_read_any
  }

  // Future variants - return future for explicit wait control via amdgcn.wait
  func.func private @load_from_global_dword_future(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_read_any {
    %transfer_size = arith.constant 4 : index
    %future = func.call @load_from_global_impl(%pos_desc, %transfer_size)
      : (!tensor_position_descriptor_2d, index) -> !future_global_read_any
    return %future : !future_global_read_any
  }

  func.func private @load_from_global_dwordx2_future(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_read_any {
    %transfer_size = arith.constant 8 : index
    %future = func.call @load_from_global_impl(%pos_desc, %transfer_size)
      : (!tensor_position_descriptor_2d, index) -> !future_global_read_any
    return %future : !future_global_read_any
  }

  func.func private @load_from_global_dwordx3_future(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_read_any {
    %transfer_size = arith.constant 12 : index
    %future = func.call @load_from_global_impl(%pos_desc, %transfer_size)
      : (!tensor_position_descriptor_2d, index) -> !future_global_read_any
    return %future : !future_global_read_any
  }

  func.func private @load_from_global_dwordx4_future(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_read_any {
    %transfer_size = arith.constant 16 : index
    %future = func.call @load_from_global_impl(%pos_desc, %transfer_size)
      : (!tensor_position_descriptor_2d, index) -> !future_global_read_any
    return %future : !future_global_read_any
  }

  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @load_from_global_dword_wait(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !v {
    %future = func.call @load_from_global_dword_future(%pos_desc)
      : (!tensor_position_descriptor_2d) -> !future_global_read_any
    %loaded_any = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded_any : !v
    return %res : !v
  }

  func.func private @load_from_global_dwordx2_wait(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !vx2 {
    %future = func.call @load_from_global_dwordx2_future(%pos_desc)
      : (!tensor_position_descriptor_2d) -> !future_global_read_any
    %loaded_any = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded_any : !vx2
    return %res : !vx2
  }

  func.func private @load_from_global_dwordx3_wait(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !vx3 {
    %future = func.call @load_from_global_dwordx3_future(%pos_desc)
      : (!tensor_position_descriptor_2d) -> !future_global_read_any
    %loaded_any = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded_any : !vx3
    return %res : !vx3
  }

  func.func private @load_from_global_dwordx4_wait(
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !vx4 {
    %future = func.call @load_from_global_dwordx4_future(%pos_desc)
      : (!tensor_position_descriptor_2d) -> !future_global_read_any
    %loaded_any = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded_any : !vx4
    return %res : !vx4
  }

  //===--------------------------------------------------------------------===//
  // Global loads 2-level 2d tiles w/ internal reshape
  //   128xf16, 256xf16, 384xf16, 512xf16 via dword, dwordx2, dwordx3, dwordx4
  // (wait, nowait and future variants)
  //===--------------------------------------------------------------------===//
  // Loads from global memory to VGPRs.
  // This function cooperatively (loads wave_size * transfer_size / elt_size)
  // elements arranged in a rows x cols matrix where num_rows is configurable.
  //
  // The actual AMDGCN instruction used is selected based on the transfer_size.
  // num_cols is computed as wave_size ceildiv num_rows.
  //
  // For example, with 64 threads per wave, f16 elements and dwordx2 transfers,
  // we have:
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
  // TODO: also add a variant with upper bounds and buffer_load to handle boundary conditions.
   func.func private @global_load_wave_elt_2d_impl(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !future_global_read_any {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "mm_pos", "nn_pos", "elt_size"] : !tensor_position_descriptor_2level_2d -> !sx2, index, index, index, index, index, index
    %num_rows, %transfer_size, %wave_size = aster_utils.struct_extract %transfer_desc ["num_rows", "transfer_size", "wave_size"] : !transfer_descriptor_2d -> index, index, index

    // static assert that %mod is 0
    %mod = affine.apply affine_map<()[wave_size, num_rows]
      -> (wave_size mod num_rows)>()[%wave_size, %num_rows]
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
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mmm_pos, %nnn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    %nnn_pos = affine.apply affine_map<()[nnn, transfer_size, elt_size] ->
      (nnn * transfer_size ceildiv elt_size)>()[%nnn, %transfer_size, %elt_size]

    // Calculate global offset
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %mmm_pos, %nnn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index, index, index) -> !index_descriptor_3level_2d
    %off_reg = func.call @tiledx2_matrix_offset(%desc) : (!index_descriptor_3level_2d) -> !v

    // Perform the load and return future (value + token)
    %res = scf.index_switch %transfer_size -> !future_global_read_any
    case 4 {
        %dst = func.call @alloc_vgprx1() : () -> (!vx1)
        %loaded, %tok_load = amdgcn.load global_load_dword dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx1) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx1
        %future = aster_utils.struct_create(%any, %tok_load) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
        scf.yield %future : !future_global_read_any
    }
    case 8 {
        %dst = func.call @alloc_vgprx2() : () -> (!vx2)
        %loaded, %tok_load = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx2) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx2
        %future = aster_utils.struct_create(%any, %tok_load) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
        scf.yield %future : !future_global_read_any
    }
    case 12 {
        %dst = func.call @alloc_vgprx3() : () -> (!vx3)
        %loaded, %tok_load = amdgcn.load global_load_dwordx3 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx3) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx3
        %future = aster_utils.struct_create(%any, %tok_load) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
        scf.yield %future : !future_global_read_any
    }
    case 16 {
        %dst = func.call @alloc_vgprx4() : () -> (!vx4)
        %loaded, %tok_load = amdgcn.load global_load_dwordx4 dest %dst addr %ptr offset d(%off_reg)
          : dps(!vx4) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %any = aster_utils.to_any %loaded : !vx4
        %future = aster_utils.struct_create(%any, %tok_load) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
        scf.yield %future : !future_global_read_any
    }
    default {
        amdgcn.sopp.sopp #amdgcn.inst<s_trap>, imm = 43
        %c0 = arith.constant 0 : index
        %any = aster_utils.to_any %c0 : index
        // Create a dummy token for the error case
        %dummy_dst = func.call @alloc_vgprx1() : () -> (!vx1)
        %dummy_loaded, %dummy_tok = amdgcn.load global_load_dword dest %dummy_dst addr %ptr offset d(%off_reg)
          : dps(!vx1) ins(!sx2, !v) -> !amdgcn.read_token<flat>
        %future = aster_utils.struct_create(%any, %dummy_tok) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read_any
        scf.yield %future : !future_global_read_any
    }

    return %res : !future_global_read_any
  }

  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @global_load_wave_128xf16_via_dword_wait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx1 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded : !vx1
    return %res : !vx1
  }

  func.func private @global_load_wave_256xf16_via_dwordx2_wait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx2 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded : !vx2
    return %res : !vx2
  }

  func.func private @global_load_wave_384xf16_via_dwordx3_wait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx3 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded : !vx3
    return %res : !vx3
  }

  func.func private @global_load_wave_512xf16_via_dwordx4_wait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx4 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_global_read_any -> !aster_utils.any, !amdgcn.read_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    %res = aster_utils.from_any %loaded : !vx4
    return %res : !vx4
  }

  // Token-returning variants for explicit wait control.
  // These return the future directly, allowing callers to use amdgcn.wait.
  func.func private @global_load_wave_128xf16_via_dword_future(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !future_global_read_any {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    return %future : !future_global_read_any
  }

  func.func private @global_load_wave_256xf16_via_dwordx2_future(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !future_global_read_any {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    return %future : !future_global_read_any
  }

  func.func private @global_load_wave_384xf16_via_dwordx3_future(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !future_global_read_any {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    return %future : !future_global_read_any
  }

  func.func private @global_load_wave_512xf16_via_dwordx4_future(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !future_global_read_any {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    return %future : !future_global_read_any
  }

  // Legacy nowait variants (return value only, token discarded).
  // Prefer _future variants for explicit wait control.
  func.func private @global_load_wave_128xf16_via_dword_nowait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx1 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any

    %res = aster_utils.from_any %loaded : !vx1
    return %res : !vx1
  }

  func.func private @global_load_wave_256xf16_via_dwordx2_nowait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx2 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any

    %res = aster_utils.from_any %loaded : !vx2
    return %res : !vx2
  }

  func.func private @global_load_wave_384xf16_via_dwordx3_nowait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx3 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any

    %res = aster_utils.from_any %loaded : !vx3
    return %res : !vx3
  }

  func.func private @global_load_wave_512xf16_via_dwordx4_nowait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d
  ) -> !vx4 {
    %future = func.call @global_load_wave_elt_2d_impl(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!future_global_read_any)
    %loaded = aster_utils.struct_extract %future ["value"] : !future_global_read_any -> !aster_utils.any

    %res = aster_utils.from_any %loaded : !vx4
    return %res : !vx4
  }

  //===--------------------------------------------------------------------===//
  // Global stores, single dword/dwordx2/dwordx3/dwordx4 (future + wait variants)
  //===--------------------------------------------------------------------===//
  // Store to global memory implementation returning a future.
  // Supports dword (4 bytes), dwordx2 (8 bytes), dwordx3 (12 bytes), and
  // dwordx4 (16 bytes) transfers.
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos (and make them workgroup/wave/thread/lane-dependent).
  // Returns a future containing the write token for explicit wait control.
  func.func private @store_to_global_impl(
    %value: !aster_utils.any,       // Value to store (v, vx2, vx3, or vx4)
    %pos_desc: !tensor_position_descriptor_2d,
    %transfer_size: index           // Transfer size in bytes (4, 8, 12, or 16)
  ) -> !future_global_write {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "elt_size"] : !tensor_position_descriptor_2d -> !sx2, index, index, index, index
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %transfer_size) : (index, index, index, index) -> !index_descriptor_2d
    %off_reg = func.call @matrix_offset(%desc) : (!index_descriptor_2d) -> !v
    %c0_store = arith.constant 0 : i32

    %res = scf.index_switch %transfer_size -> !future_global_write
    case 4 {
      %data = aster_utils.from_any %value : !v
      %token = amdgcn.store global_store_dword data %data addr %ptr offset d(%off_reg) + c(%c0_store)
        : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>
      scf.yield %token : !future_global_write
    }
    case 8 {
      %data = aster_utils.from_any %value : !vx2
      %token = amdgcn.store global_store_dwordx2 data %data addr %ptr offset d(%off_reg) + c(%c0_store)
        : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>
      scf.yield %token : !future_global_write
    }
    case 12 {
      %data = aster_utils.from_any %value : !vx3
      %token = amdgcn.store global_store_dwordx3 data %data addr %ptr offset d(%off_reg) + c(%c0_store)
        : ins(!vx3, !sx2, !v, i32) -> !amdgcn.write_token<flat>
      scf.yield %token : !future_global_write
    }
    case 16 {
      %data = aster_utils.from_any %value : !vx4
      %token = amdgcn.store global_store_dwordx4 data %data addr %ptr offset d(%off_reg) + c(%c0_store)
        : ins(!vx4, !sx2, !v, i32) -> !amdgcn.write_token<flat>
      scf.yield %token : !future_global_write
    }
    default {
      // Note: this is an unexpected path needed for completeness, it will trap.
      %future = func.call @trapping_undef_future_global_write() : () -> !future_global_write
      scf.yield %future : !future_global_write
    }

    return %res : !future_global_write
  }

  // Future variants - return future for explicit wait control via amdgcn.wait
  func.func private @store_to_global_dword_future(
    %value: !v,
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_write {
    %transfer_size = arith.constant 4 : index
    %any_value = aster_utils.to_any %value : !v
    %future = func.call @store_to_global_impl(%any_value, %pos_desc, %transfer_size)
      : (!aster_utils.any, !tensor_position_descriptor_2d, index) -> !future_global_write
    return %future : !future_global_write
  }

  func.func private @store_to_global_dwordx2_future(
    %value: !vx2,
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_write {
    %transfer_size = arith.constant 8 : index
    %any_value = aster_utils.to_any %value : !vx2
    %future = func.call @store_to_global_impl(%any_value, %pos_desc, %transfer_size)
      : (!aster_utils.any, !tensor_position_descriptor_2d, index) -> !future_global_write
    return %future : !future_global_write
  }

  func.func private @store_to_global_dwordx3_future(
    %value: !vx3,
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_write {
    %transfer_size = arith.constant 12 : index
    %any_value = aster_utils.to_any %value : !vx3
    %future = func.call @store_to_global_impl(%any_value, %pos_desc, %transfer_size)
      : (!aster_utils.any, !tensor_position_descriptor_2d, index) -> !future_global_write
    return %future : !future_global_write
  }

  func.func private @store_to_global_dwordx4_future(
    %value: !vx4,
    %pos_desc: !tensor_position_descriptor_2d
  ) -> !future_global_write {
    %transfer_size = arith.constant 16 : index
    %any_value = aster_utils.to_any %value : !vx4
    %future = func.call @store_to_global_impl(%any_value, %pos_desc, %transfer_size)
      : (!aster_utils.any, !tensor_position_descriptor_2d, index) -> !future_global_write
    return %future : !future_global_write
  }

  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @store_to_global_dword_wait(
    %value: !v,
    %pos_desc: !tensor_position_descriptor_2d
  ) {
    %future = func.call @store_to_global_dword_future(%value, %pos_desc)
      : (!v, !tensor_position_descriptor_2d) -> !future_global_write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  func.func private @store_to_global_dwordx2_wait(
    %value: !vx2,
    %pos_desc: !tensor_position_descriptor_2d
  ) {
    %future = func.call @store_to_global_dwordx2_future(%value, %pos_desc)
      : (!vx2, !tensor_position_descriptor_2d) -> !future_global_write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  func.func private @store_to_global_dwordx3_wait(
    %value: !vx3,
    %pos_desc: !tensor_position_descriptor_2d
  ) {
    %future = func.call @store_to_global_dwordx3_future(%value, %pos_desc)
      : (!vx3, !tensor_position_descriptor_2d) -> !future_global_write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  func.func private @store_to_global_dwordx4_wait(
    %value: !vx4,
    %pos_desc: !tensor_position_descriptor_2d
  ) {
    %future = func.call @store_to_global_dwordx4_future(%value, %pos_desc)
      : (!vx4, !tensor_position_descriptor_2d) -> !future_global_write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }


  //===--------------------------------------------------------------------===//
  // MFMA fragment A ds_read 2-level 2d tiles w/ internal reshape
  //   16x16xf16 via ds_read_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Read the `A` fragment (16x16xf16) from LDS to VGPRs, returning a future.
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_read_A_wave_16x16xf16_fragment_impl(
    %pos_desc: !lds_position_descriptor_2d,
    %transposed: i1              // Whether to transpose the indexing
  ) -> !future_lds_read_any {
    %lds_base, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index
    // Compute the MFMA positions
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %mm_pos_raw, %nn_pos_raw = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index
    %mm_pos, %nn_pos = scf.if %transposed -> (index, index) {
      scf.yield %nn_pos_raw, %mm_pos_raw : index, index
    } else {
      scf.yield %mm_pos_raw, %nn_pos_raw : index, index
    }
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the DS read and return future
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds_reg offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %any = aster_utils.to_any %from_lds : !vx2
    %future = aster_utils.struct_create(%any, %tok_read) : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read_any

    return %future : !future_lds_read_any
  }

  // Read the `A` fragment (16x16xf16) from LDS to VGPRs, in a **synchronized
  // fashion** (i.e. waitcnt 0 is inserted after the ds_read).
  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(
    %pos_desc: !lds_position_descriptor_2d,
    %transposed: i1
  ) -> !vx2 {
    %future = func.call @lds_read_A_wave_16x16xf16_fragment_impl(%pos_desc, %transposed) : (!lds_position_descriptor_2d, i1) -> !future_lds_read_any
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %res = aster_utils.from_any %loaded : !vx2
    return %res : !vx2
  }

  // Token-returning variant for explicit wait control.
  // Returns the future directly, allowing callers to use amdgcn.wait.
  func.func private @lds_read_A_wave_16x16xf16_fragment_future(
    %pos_desc: !lds_position_descriptor_2d,
    %transposed: i1
  ) -> !future_lds_read_any {
    %future = func.call @lds_read_A_wave_16x16xf16_fragment_impl(%pos_desc, %transposed) : (!lds_position_descriptor_2d, i1) -> !future_lds_read_any
    return %future : !future_lds_read_any
  }


  //===--------------------------------------------------------------------===//
  // Swizzled MFMA fragment A ds_read 2-level 2d tiles w/ internal reshape
  //   16x16xf16 via ds_read_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Read the `A` fragment (16x16xf16) from LDS to VGPRs with swizzling, returning a future.
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_impl(
    %pos_desc: !lds_position_descriptor_2d
  ) -> !future_lds_read_any {
    %lds_base, %m_pos, %n_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["lds_base", "m_pos", "n_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2d -> index, index, index, index, index

    // Apply A-matrix swizzle
    %mfma_idx_A = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %swizzled_idx = func.call @xor_swizzled_mfma_index_16xf16(%mfma_idx_A) : (!index_pair) -> !index_pair
    %swizzled_row, %swizzled_col = aster_utils.struct_extract %swizzled_idx ["i", "j"] : !index_pair -> index, index
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %swizzled_row, %swizzled_col, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %result, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    %any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%any, %tok_read) : (!aster_utils.any, !amdgcn.read_token<shared>) -> !future_lds_read_any

    return %future : !future_lds_read_any
  }

  // Read the `A` fragment (16x16xf16) from LDS to VGPRs with swizzling, in a
  // **synchronized fashion** (i.e. waitcnt 0 is inserted after the ds_read).
  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(
    %pos_desc: !lds_position_descriptor_2d
  ) -> !vx2 {
    %future = func.call @lds_read_swizzled_wave_16x16xf16_fragment_impl(%pos_desc) : (!lds_position_descriptor_2d) -> !future_lds_read_any
    %loaded, %token = aster_utils.struct_extract %future ["value", "token"] : !future_lds_read_any -> !aster_utils.any, !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %res = aster_utils.from_any %loaded : !vx2
    return %res : !vx2
  }

  // Token-returning variant for explicit wait control.
  // Returns the future directly, allowing callers to use amdgcn.wait.
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_future(
    %pos_desc: !lds_position_descriptor_2d
  ) -> !future_lds_read_any {
    %future = func.call @lds_read_swizzled_wave_16x16xf16_fragment_impl(%pos_desc) : (!lds_position_descriptor_2d) -> !future_lds_read_any
    return %future : !future_lds_read_any
  }

  //===--------------------------------------------------------------------===//
  // MFMA fragment C global_store 2-level 2d tiles w/ internal reshape
  //   16x16xf32 via dwordx4 (transposed variant) or dwordx4 (non-transposed variant)
  // (wait variants)
  //===--------------------------------------------------------------------===//
  // Store the `C` fragment (16x16xf32) from VGPRs to global memory, in a
  // **synchronized fashion** (i.e. waitcnt 0 is inserted after each global_store).
  // The caller is responsible for embedding distribution information into the
  // positions. The callee computes and embeds the MFMA positions.
  // This function assumes a major/minor tile structure for the global positions.
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(
    %acc: !vx4,                     // The accumulator fragment to store
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %transposed: i1                 // Whether to transpose the indexing
  ) {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "mm_pos", "nn_pos", "elt_size"] : !tensor_position_descriptor_2level_2d -> !sx2, index, index, index, index, index, index
    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    scf.if %transposed {
      // Split the fragment into 4 dword values
      %v0, %v1, %v2, %v3 = amdgcn.split_register_range %acc : !vx4
      %C_fragment =  memref.alloca() : memref<4x!v>
      memref.store %v0, %C_fragment[%c0] : memref<4x!v>
      memref.store %v1, %C_fragment[%c1] : memref<4x!v>
      memref.store %v2, %C_fragment[%c2] : memref<4x!v>
      memref.store %v3, %C_fragment[%c3] : memref<4x!v>

      // Compute the transposed MFMA positions.
      %mfma_idx_C = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
      %nnn_pos, %mmm_pos = aster_utils.struct_extract %mfma_idx_C ["i", "j"] : !index_pair -> index, index

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

        // Create position descriptor
        %pos_desc_2d = aster_utils.struct_create(%ptr, %m_global_pos, %n_global_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        // Store to global memory with wait
        func.call @store_to_global_dword_wait(%fragment, %pos_desc_2d)
          : (!v, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } else {
      // Compute the MFMA positions
      %mfma_idx_C2 = func.call @mfma_index_C_16x16xf32() : () -> !index_pair
      %mmm_pos, %nnn_pos = aster_utils.struct_extract %mfma_idx_C2 ["i", "j"] : !index_pair -> index, index

      // Calculate global j position
      %m_global_pos = affine.apply
        affine_map<()[m_pos, mm_pos, mmm_pos] -> (m_pos + mm_pos + mmm_pos)>
        ()[%m_pos, %mm_pos, %mmm_pos]
      %n_global_pos_in_f32 = affine.apply
        affine_map<()[n_pos, nn_pos, nnn_pos] -> (n_pos + nn_pos + nnn_pos)>
        ()[%n_pos, %nn_pos, %nnn_pos]
      // Translate n in units of the transfer size (dwordx4).
      %n_global_pos = affine.apply affine_map<()[n_global_pos_in_f32]
        -> (n_global_pos_in_f32 floordiv 4)>()[%n_global_pos_in_f32]

      // Create position descriptor
      %pos_desc_2d = aster_utils.struct_create(%ptr, %m_global_pos, %n_global_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
      // Store to global memory with wait
      func.call @store_to_global_dwordx4_wait(%acc, %pos_desc_2d)
        : (!vx4, !tensor_position_descriptor_2d) -> ()
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // LDS write 2-level 2d tiles w/ internal reshape
  //   256xf16 via ds_write_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Writes %value to LDS, returning a future for explicit wait control.
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
  func.func private @lds_write_wave_256xf16_via_dwordx2_impl(
    %pos_desc: !lds_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d,
    %value: !vx2                 // The value to write to LDS
  ) -> !future_lds_write {
    %lds_base_off, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size = aster_utils.struct_extract %pos_desc ["lds_base", "mm_pos", "nn_pos", "lds_stride_in_bytes", "elt_size"] : !lds_position_descriptor_2level_2d -> index, index, index, index, index
    %num_rows, %transfer_size, %wave_size = aster_utils.struct_extract %transfer_desc ["num_rows", "transfer_size", "wave_size"] : !transfer_descriptor_2d -> index, index, index

    %num_cols = affine.apply affine_map<()[wave_size, num_rows]
      -> (wave_size ceildiv num_rows)>()[%wave_size, %num_rows]

    // Get local positions within the minor tile
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mmm_pos, %nnn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    %nnn_pos = affine.apply affine_map<()[nnn, transfer_size, elt_size] ->
      (nnn * transfer_size ceildiv elt_size)>()[%nnn, %transfer_size, %elt_size]

    // Calculate offset into LDS
    %desc = aster_utils.struct_create(%mm_pos, %nn_pos, %mmm_pos, %nnn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // DS write to LDS and return token
    %l_off_i32 = arith.index_cast %lds_base_off : index to i32
    %token = amdgcn.store ds_write_b64 data %value addr %off_lds_reg offset c(%l_off_i32) : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    return %token : !future_lds_write
  }

  // Writes %value to LDS, in a **synchronized fashion** (i.e. waitcnt 0 is
  // inserted after ds_write).
  // Wait variants - call future variant and wait via s_waitcnt (no amdgcn-convert-waits for now)
  // TODO: use only amdgcn-convert-waits pass
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(
    %pos_desc: !lds_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d,
    %value: !vx2
  ) {
    %_token = func.call @lds_write_wave_256xf16_via_dwordx2_impl(%pos_desc, %transfer_desc, %value) : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return
  }

  // Token-returning variant for explicit wait control.
  // Returns the future directly, allowing callers to use amdgcn.wait.
  func.func private @lds_write_wave_256xf16_via_dwordx2_future(
    %pos_desc: !lds_position_descriptor_2level_2d,
    %transfer_desc: !transfer_descriptor_2d,
    %value: !vx2
  ) -> !future_lds_write {
    %future = func.call @lds_write_wave_256xf16_via_dwordx2_impl(%pos_desc, %transfer_desc, %value) : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write
    return %future : !future_lds_write
  }

  //===--------------------------------------------------------------------===//
  // Global load to LDS write 2-level 2d tiles w/ internal reshape
  //   256xf16 via ds_write_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Load a 16x16xf16 tile from global memory to LDS within a single wave, in
  // a **synchronized fashion** (i.e. waitcnt 0 are inserted after global_load
  // and after lds_write).
  // This function cooperatively loads a 16x16xf16 tile from global memory to LDS
  // and forces num_rows to be 16, resulting in non-coalesced accesses in order
  // to preserve the 16x16 shape.
  // TODO: support different global_load and lds_write num_rows, to achieve a
  // reshape (only when we have a clear use for it).
  func.func private @global_load_to_lds_wave_16x16_f16_wait(
    %pos_desc: !tensor_position_descriptor_2level_2d,
    %lds_base_off: index,           // The local base offset in LDS
    %LDS_STRIDE_IN_BYTES: index     // The inner-most major-tile size **in bytes** in LDS
  ) {
    %ptr, %m_pos, %n_pos, %GLOBAL_STRIDE_IN_BYTES, %mm_pos, %nn_pos, %elt_size = aster_utils.struct_extract %pos_desc ["ptr", "m_pos", "n_pos", "global_stride_in_bytes", "mm_pos", "nn_pos", "elt_size"] : !tensor_position_descriptor_2level_2d -> !sx2, index, index, index, index, index, index
    %num_rows = arith.constant 16 : index
    %transfer_size = arith.constant 8 : index // dwordx2 size in bytes
    %wave_size = arith.constant 64 : index    // 64 threads per wave
    %transfer_desc = aster_utils.struct_create(%num_rows, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
    %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc, %transfer_desc) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
    %lds_pos_desc = aster_utils.struct_create(%lds_base_off, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
    func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc, %transfer_desc, %loaded)
      : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
    return
  }

}
