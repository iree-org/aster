// Kittens global memory transfer primitives for 16x64 bytes tiles (dwordx4 loads).
//
// Each thread loads 16 bytes per dwordx4 load.
// 64 threads x 16 bytes/thread = 1024 bytes = full 16x64_b tile per load.
//
// Uses buffer_load_dwordx4 (MUBUF OFFEN mode) with 4-SGPR buffer descriptors
// for 3-component addressing: soffset(SGPR) + voffset(VGPR) + inst_offset(imm).

// Register types
!s = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx4 = !amdgcn.sgpr<[? + 4]>
!v = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr<[? + 4]>

// Future type
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.library @kittens_global_16x64_b isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @alloc_vgprx4() -> !vx4
  // From indexing.mlir
  func.func private @thread_tile_pos_16x64_b() -> (index, index)
  func.func private @tiled_row_byte_off(index, index, index, index, index, index) -> index
  // From indexing_ptr.mlir
  func.func private @index_to_vgpr_i32(index) -> !v

  // Compute the byte offset for a 16x64_b global tile load.
  // Returns index (byte offset) that can be placed anywhere in the schedule.
  func.func private @compute_global_byte_off_16x64_b(
      %m: index, %n: index, %stride: index
  ) -> index {
    %row, %col_byte = func.call @thread_tile_pos_16x64_b() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %byte_off = func.call @tiled_row_byte_off(%m, %row, %n, %col_byte, %stride, %elt_size)
        : (index, index, index, index, index, index) -> index
    return %byte_off : index
  }

  // Issue dwordx4 buffer load at a pre-computed byte offset.
  // Uses MUBUF OFFEN mode: SRD base + voffset (all offset in VGPR).
  func.func private @load_global_at_byte_off(
      %buffer_resource: !aster_utils.any, %byte_off: index
  ) -> !future_global_read {
    %buffer_resource_sx4 = aster_utils.from_any %buffer_resource : !sx4
    %voffset = func.call @index_to_vgpr_i32(%byte_off) : (index) -> !v

    // soffset = 0 (all offset in voffset for now; optimizer can split later)
    %c0 = arith.constant 0 : i32
    %soffset = lsir.to_reg %c0 : i32 -> !s

    %tmp_reg = func.call @alloc_vgprx4() : () -> !vx4
    %loaded, %tok_global = amdgcn.buffer_load_dwordx4 dest %tmp_reg addr %buffer_resource_sx4 offset u(%soffset) + off_idx(%voffset) + c(%c0) {offen} : outs(!vx4) ins(!sx4, !s, !v) mods(i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %loaded : !vx4
    %future = aster_utils.struct_create(%value_any, %tok_global)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  // Convenience wrapper: compute byte offset + issue load in one call.
  func.func private @load_global_tile_16x64_b(
      %buffer_resource: !aster_utils.any, %m: index, %n: index, %stride: index
  ) -> !future_global_read {
    %byte_off = func.call @compute_global_byte_off_16x64_b(%m, %n, %stride)
        : (index, index, index) -> index
    %future = func.call @load_global_at_byte_off(%buffer_resource, %byte_off)
        : (!aster_utils.any, index) -> !future_global_read
    return %future : !future_global_read
  }

  // Construct !sx4 buffer resource from raw !sx2 kernel arg, then type-erase.
  // From indexing_ptr.mlir
  func.func private @make_raw_buffer_rsrc(!sx2) -> !sx4

  func.func private @prepare_ptr(%raw: !sx2) -> !aster_utils.any {
    %buffer_resource = func.call @make_raw_buffer_rsrc(%raw) : (!sx2) -> !sx4
    %erased = aster_utils.to_any %buffer_resource : !sx4
    return %erased : !aster_utils.any
  }

  // Note: store_global depends on the shape / layout after computation, see
  // store_global_C_mfma_f32_16x16x16_f16 in compute_16x16_f16.mlir.
}
