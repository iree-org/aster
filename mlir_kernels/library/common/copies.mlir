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


//===-----------------------------------------------------------------------===//
// Single thread, single global load / store instruction, parameterizable
// by !tensor_position_descriptor_2d and transfer size (4, 8, 12, or 16 bytes).
//===-----------------------------------------------------------------------===//
amdgcn.library @single_global_load_store_to_vgpr_single_thread isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @matrix_offset(!index_descriptor_2d) -> !v
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

}

//===-----------------------------------------------------------------------===//
// Wave-level, single global load (TODO: store) instruction, parameterizable by
// !tensor_position_descriptor_2level_2d and !transfer_descriptor_2d.
//
// Allows moving 128xf16, 256xf16, 384xf16, 512xf16 via (resp.) dword, x2, x3, x4
// as a 2-D matrix with num_rows defined by !transfer_descriptor_2d.
//===-----------------------------------------------------------------------===//
amdgcn.library @single_global_load_store_to_vgpr_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // From above library.
  func.func private @store_to_global_dword_wait(%value: !v, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx2_wait(%value: !vx2, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx3_wait(%value: !vx3, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx4_wait(%value: !vx4, %pos_desc: !tensor_position_descriptor_2d) -> ()

  //===--------------------------------------------------------------------===//
  // Global loads 2-level 2d tiles w/ internal reshape
  //   128xf16, 256xf16, 384xf16, 512xf16 via dword, dwordx2, dwordx3, dwordx4
  // (wait, nowait and future variants)
  //===--------------------------------------------------------------------===//
  // Loads from global memory to VGPRs cooperatively across a wave.
  // Total elements loaded = wave_size * transfer_size / elt_size, arranged in
  // a num_rows x num_cols matrix where num_cols = wave_size / num_rows.
  //
  // Parameters come from two descriptors:
  //
  //   !tensor_position_descriptor_2level_2d (position in memory):
  //     - ptr: base pointer
  //     - m_pos, n_pos: major tile position (element coordinates)
  //     - mm_pos, nn_pos: minor tile position within major tile
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes
  //
  //   !transfer_descriptor_2d (transfer configuration):
  //     - num_rows: number of rows in the cooperative load pattern
  //     - transfer_size: bytes per thread (4=dword, 8=dwordx2, 12=dwordx3, 16=dwordx4)
  //     - wave_size: threads per wave (typically 64)
  //
  // The num_rows parameter controls how threads are distributed across the tile:
  //   num_cols = wave_size / num_rows
  //   Each thread loads transfer_size bytes at position (lane / num_cols, lane % num_cols)
  //
  // Example: wave_size=64, elt_size=2 (f16), transfer_size=8 (dwordx2):
  //   num_rows= 1: 1x64 threads, each loads 4xf16 → tile is  1x256xf16
  //   num_rows=16: 16x4 threads, each loads 4xf16 → tile is 16x16xf16
  //   num_rows=64: 64x1 threads, each loads 4xf16 → tile is 64x4xf16
  //
  // Choosing num_rows for coalescing:
  //   - num_rows=1: best when global_stride_in_bytes >= wave_size * transfer_size
  //     (all threads access contiguous memory in one row)
  //   - num_rows=16: good for 16x16 tiles typical in MFMA workloads
  //   - Higher num_rows: useful when stride is small or for pipelining flexibility
  //
  // Note: positions (m_pos, n_pos, mm_pos, nn_pos) are in element counts, not bytes.
  //
  // TODO: add variant with upper bounds and buffer_load for boundary conditions.
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
}

//===-----------------------------------------------------------------------===//
// Wave-level, single LDS read instruction for MFMA fragment layouts,
// parameterizable by !lds_position_descriptor_2d and %transposed flag.
//
// Reads 16x16xf16 tiles via ds_read_b64 into MFMA "A" fragment layout.
// Thread mapping follows @mfma_index_A_16x16xf16(); %transposed swaps row/col.
//===-----------------------------------------------------------------------===//
amdgcn.library @single_lds_read_mfma_fragment_to_vgpr_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // From above library.
  func.func private @store_to_global_dword_wait(%value: !v, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx2_wait(%value: !vx2, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx3_wait(%value: !vx3, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx4_wait(%value: !vx4, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc: !tensor_position_descriptor_2level_2d, %transfer_desc: !transfer_descriptor_2d) -> !vx2

  //===--------------------------------------------------------------------===//
  // MFMA fragment A ds_read 2-level 2d tiles w/ internal reshape
  //   16x16xf16 via ds_read_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Reads a 16x16xf16 tile from LDS into MFMA "A" fragment layout in VGPRs.
  // Each thread reads 4xf16 (8 bytes) via ds_read_b64, totaling 256 f16 elements
  // across 64 threads.
  //
  // Parameters from !lds_position_descriptor_2d:
  //   - lds_base: base offset in LDS (bytes)
  //   - m_pos, n_pos: tile position in elements (row, col)
  //   - lds_stride_in_bytes: row stride in LDS
  //   - elt_size: element size in bytes (2 for f16)
  //
  // The %transposed parameter swaps the MFMA indexing pattern:
  //   - false: standard A matrix layout (row-major access)
  //   - true: transposed layout (column-major access, for B matrix as A^T)
  //
  // Thread mapping follows MFMA 16x16 layout from @mfma_index_A_16x16xf16():
  //   Each thread's (mm_pos, nn_pos) determines its 4xf16 slice within the tile.
  //
  // Returns a future containing the loaded !vx2 value and read token.
  func.func private @lds_read_A_wave_16x16xf16_fragment_impl(
    %pos_desc: !lds_position_descriptor_2d, %transposed: i1
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

}

//===-----------------------------------------------------------------------===//
// Wave-level, single LDS read instruction for swizzled MFMA fragment layouts,
// parameterizable by !lds_position_descriptor_2d.
//
// Reads 16x16xf16 tiles via ds_read_b64 with XOR swizzling to reduce bank conflicts.
// Thread mapping: @mfma_index_A_16x16xf16() → @xor_swizzled_mfma_index_16xf16().
//===-----------------------------------------------------------------------===//
amdgcn.library @single_lds_read_swizzled_mfma_fragment_to_vgpr_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // From above library.
  func.func private @store_to_global_dword_wait(%value: !v, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx2_wait(%value: !vx2, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx3_wait(%value: !vx3, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx4_wait(%value: !vx4, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc: !tensor_position_descriptor_2level_2d, %transfer_desc: !transfer_descriptor_2d) -> !vx2

  //===--------------------------------------------------------------------===//
  // Swizzled MFMA fragment A ds_read 2-level 2d tiles w/ internal reshape
  //   16x16xf16 via ds_read_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Reads a 16x16xf16 tile from LDS into MFMA "A" fragment layout with XOR swizzling.
  // Each thread reads 4xf16 (8 bytes) via ds_read_b64, totaling 256 f16 elements
  // across 64 threads.
  //
  // Parameters from !lds_position_descriptor_2d:
  //   - lds_base: base offset in LDS (bytes)
  //   - m_pos, n_pos: tile position in elements (row, col)
  //   - lds_stride_in_bytes: row stride in LDS
  //   - elt_size: element size in bytes (2 for f16)
  //
  // Swizzling is applied via @xor_swizzled_mfma_index_16xf16() to reduce LDS
  // bank conflicts. The swizzle pattern XORs row bits into column bits, spreading
  // thread accesses across different banks.
  //
  // Thread mapping: @mfma_index_A_16x16xf16() → @xor_swizzled_mfma_index_16xf16()
  //   Each thread's swizzled (row, col) determines its 4xf16 slice within the tile.
  //
  // Returns a future containing the loaded !vx2 value and read token.
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
}

//===-----------------------------------------------------------------------===//
// Wave-level, single LDS write instruction, parameterizable by
// !lds_position_descriptor_2level_2d and !transfer_descriptor_2d.
//
// Writes 256xf16 tiles via ds_write_b64 as a 2-D matrix with num_rows
// defined by !transfer_descriptor_2d (typically matches producer global load).
//===-----------------------------------------------------------------------===//
amdgcn.library @single_lds_write_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // From above library.
  func.func private @store_to_global_dword_wait(%value: !v, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx2_wait(%value: !vx2, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx3_wait(%value: !vx3, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx4_wait(%value: !vx4, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc: !tensor_position_descriptor_2level_2d, %transfer_desc: !transfer_descriptor_2d) -> !vx2

  //===--------------------------------------------------------------------===//
  // LDS write 2-level 2d tiles w/ internal reshape
  //   256xf16 via ds_write_b64
  // (wait and future variants)
  //===--------------------------------------------------------------------===//
  // Writes %value to LDS cooperatively across a wave, returning a write token.
  // Total elements written = wave_size * transfer_size / elt_size, arranged in
  // a num_rows x num_cols matrix where num_cols = wave_size / num_rows.
  //
  // Parameters come from two descriptors:
  //
  //   !lds_position_descriptor_2level_2d (position in LDS):
  //     - lds_base: base offset in LDS (bytes)
  //     - mm_pos, nn_pos: minor tile position (element coordinates)
  //     - lds_stride_in_bytes: row stride
  //     - elt_size: element size in bytes
  //
  //   !transfer_descriptor_2d (transfer configuration):
  //     - num_rows: number of rows in the cooperative write pattern
  //     - transfer_size: bytes per thread (8 for ds_write_b64)
  //     - wave_size: threads per wave (typically 64)
  //
  // The num_rows parameter controls how threads are distributed across the tile:
  //   num_cols = wave_size / num_rows
  //   Each thread writes transfer_size bytes at position (lane / num_cols, lane % num_cols)
  //
  // Example: wave_size=64, elt_size=2 (f16), transfer_size=8 (dwordx2):
  //   num_rows= 1: 1x64 threads, each writes 4xf16 → tile is  1x256xf16
  //   num_rows=16: 16x4 threads, each writes 4xf16 → tile is 16x16xf16
  //   num_rows=64: 64x1 threads, each writes 4xf16 → tile is 64x4xf16
  //
  // Typically num_rows should match the producer global load to preserve layout.
  //
  // Note: positions (mm_pos, nn_pos) are in element counts, not bytes.
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
}

//===-----------------------------------------------------------------------===//
// Wave-level, global store instruction for MFMA C fragment layouts,
// parameterizable by !tensor_position_descriptor_2level_2d and %transposed flag.
//
// Stores 16x16xf32 MFMA "C" accumulator tiles via global_store_dword (4 per thread).
// Thread mapping follows @mfma_index_C_16x16xf32(); %transposed swaps row/col.
//===-----------------------------------------------------------------------===//
amdgcn.library @single_global_store_mfma_fragment_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx1() -> !vx1
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx3() -> !vx3
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @xor_swizzled_mfma_index_16xf16(!index_pair) -> !index_pair
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair
  // From above library.
  func.func private @store_to_global_dword_wait(%value: !v, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx2_wait(%value: !vx2, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx3_wait(%value: !vx3, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @store_to_global_dwordx4_wait(%value: !vx4, %pos_desc: !tensor_position_descriptor_2d) -> ()
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc: !tensor_position_descriptor_2level_2d, %transfer_desc: !transfer_descriptor_2d) -> !vx2

  //===--------------------------------------------------------------------===//
  // MFMA fragment C global_store 2-level 2d tiles w/ internal reshape
  //   16x16xf32 via global_store_dword (4 stores per thread)
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Stores a 16x16xf32 MFMA "C" accumulator fragment from VGPRs to global memory.
  // Each thread stores 4xf32 (16 bytes) via 4 separate global_store_dword ops,
  // totaling 256 f32 elements across 64 threads.
  //
  // Parameters:
  //   %acc: !vx4 - the accumulator fragment to store
  //   %pos_desc: !tensor_position_descriptor_2level_2d:
  //     - ptr: base pointer
  //     - m_pos, n_pos: major tile position (element coordinates)
  //     - mm_pos, nn_pos: minor tile position within major tile
  //     - global_stride_in_bytes: row stride
  //     - elt_size: element size in bytes (4 for f32)
  //   %transposed: i1 - swaps row/col indexing for transposed output layout
  //
  // Thread mapping follows MFMA 16x16 C layout from @mfma_index_C_16x16xf32():
  //   Each thread's (mmm_pos, nnn_pos) + loop over 4 elements determines store positions.
  //
  // Synchronization: inserts s_waitcnt vmcnt=0 after all stores complete.
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
}

//===-----------------------------------------------------------------------===//
// Wave-level, global→LDS copy instruction, parameterizable by
// !tensor_position_descriptor_2level_2d and LDS offset/stride.
//
// Copies 16x16xf16 tiles via dwordx2 global load + ds_write_b64.
// Fixed num_rows=16 layout preserving 16x16 tile shape; synchronized with waitcnt.
//===-----------------------------------------------------------------------===//
amdgcn.library @single_global_load_to_lds_single_wave isa = [#amdgcn.isa<cdna3>] {
  // From copies.mlir (@common_copies)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2)

  //===--------------------------------------------------------------------===//
  // Global load to LDS write: 16x16xf16 tile copy
  // (wait variant only)
  //===--------------------------------------------------------------------===//
  // Copies a 16x16xf16 tile from global memory to LDS within a single wave.
  // Each thread loads/stores 4xf16 (8 bytes) via dwordx2, totaling 256 f16 elements
  // across 64 threads.
  //
  // Parameters:
  //   %pos_desc: !tensor_position_descriptor_2level_2d (global memory position)
  //   %lds_base_off: base offset in LDS (bytes)
  //   %LDS_STRIDE_IN_BYTES: row stride in LDS
  //
  // Thread distribution: fixed num_rows=16, resulting in 16x4 thread grid.
  // This preserves the 16x16 tile shape but may not achieve optimal coalescing.
  //
  // Synchronization: inserts s_waitcnt vmcnt=0 after global load and lgkmcnt=0
  // after LDS write.
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
