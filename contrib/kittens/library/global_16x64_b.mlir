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
  func.func private @thread_tile_pos_16x64b() -> (index, index)
  func.func private @tiled_row_byte_off(index, index, index, index, index, index) -> index
  // From indexing_ptr.mlir
  func.func private @global_addr_from_offset(!sx2, index) -> !vx2

  // Issue dwordx4 global load for a 16x64_b tile, return future.
  func.func private @load_global_tile_16x64_b(
      %global_ptr: !sx2, %m: index, %n: index, %stride: index
  ) -> !future_global_read {
    %row, %col_byte = func.call @thread_tile_pos_16x64b() : () -> (index, index)
    %elt_size = arith.constant 2 : index
    %byte_off = func.call @tiled_row_byte_off(%m, %row, %n, %col_byte, %stride, %elt_size)
        : (index, index, index, index, index, index) -> index
    %addr = func.call @global_addr_from_offset(%global_ptr, %byte_off)
        : (!sx2, index) -> !vx2

    %tmp_reg = func.call @alloc_vgprx4() : () -> !vx4
    %loaded, %tok_global = amdgcn.load global_load_dwordx4 dest %tmp_reg addr %addr
        : dps(!vx4) ins(!vx2) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %loaded : !vx4
    %future = aster_utils.struct_create(%value_any, %tok_global)
        : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  // Note: store_global depends on the shape / layout after computation, see
  // store_global_C_mfma_f32_16x16x16_f16 in compute_16x16_f16.mlir.
}
