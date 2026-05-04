// Kittens global memory transfer primitives for 16x64 bytes tiles (dwordx4 loads).
//
// Each thread loads 16 bytes per dwordx4 load.
// 64 threads x 16 bytes/thread = 1024 bytes = full 16x64_b tile per load.

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
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
  func.func private @global_addr_from_offset(!sx2, index) -> !vx2

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

  // Issue dwordx4 global load at a pre-computed byte offset.
  func.func private @load_global_at_byte_off(
      %global_ptr: !aster_utils.any, %byte_off: index
  ) -> !future_global_read {
    %global_ptr_sx2 = aster_utils.from_any %global_ptr : !sx2
    %addr = func.call @global_addr_from_offset(%global_ptr_sx2, %byte_off)
        : (!sx2, index) -> !vx2
    %tmp_reg = func.call @alloc_vgprx4() : () -> !vx4
    %c0_i32_mig1 = arith.constant 0 : i32
    %loaded, %tok_global = amdgcn.global_load_dwordx4 dest %tmp_reg addr %addr offset c(%c0_i32_mig1) : outs(!vx4) ins(!vx2) mods(i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %loaded : !vx4
    %future = aster_utils.struct_create(%value_any, %tok_global)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  // Convenience wrapper: compute byte offset + issue load in one call.
  func.func private @load_global_tile_16x64_b(
      %global_ptr: !aster_utils.any, %m: index, %n: index, %stride: index
  ) -> !future_global_read {
    %byte_off = func.call @compute_global_byte_off_16x64_b(%m, %n, %stride)
        : (index, index, index) -> index
    %future = func.call @load_global_at_byte_off(%global_ptr, %byte_off)
        : (!aster_utils.any, index) -> !future_global_read
    return %future : !future_global_read
  }

  // Type-erase a raw !sx2 kernel arg pointer to !aster_utils.any.
  // Flat mode: no buffer resource construction needed.
  func.func private @prepare_ptr(%raw: !sx2) -> !aster_utils.any {
    %erased = aster_utils.to_any %raw : !sx2
    return %erased : !aster_utils.any
  }

  // Note: store_global depends on the shape / layout after computation, see
  // store_global_C_mfma_f32_16x16x16_f16 in compute_16x16_f16.mlir.
}
