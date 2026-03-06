// Kittens-style 32x32_f16 tile abstractions for global load/store feeding into
// MFMA 32x32x8 operations.
// Provides high-level primitives for register tiles used in GEMM kernels.
//
// Key differences from 16x16:
//   - MFMA: v_mfma_f32_32x32x8_f16 (32x32 output, K=8)
//   - A/B inputs: still 2 VGPRs (same), but layout is 32x8
//   - C/D output: 16 VGPRs (was 4)
//   - Lane mapping: lane%32 gives row (was lane%16)

// Register types from common library
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx16 = !amdgcn.vgpr<[? + 16]>

// Token types for async memory operations
!write_token = !amdgcn.write_token<flat>

// Future types from futures.mlir
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

// Kittens register tile type aliases for 32x32x8 MFMA
//   !rt_A_f16: A operand - 2 VGPRs holding 4xf16 per thread (32x8 logical tile)
//   !rt_B_f16: B operand - 2 VGPRs holding 4xf16 per thread (32x8 logical tile)
//   !rt_C_f32: C/D operand - 16 VGPRs holding 16xf32 per thread (32x32 logical tile)
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx16

amdgcn.library @kittens_global_32x32_f16 isa = [#amdgcn.isa<cdna3>] {
  // From register-init.mlir
  func.func private @init_vgprx16(i32) -> !vx16
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx16() -> !vx16
  // From indexing.mlir
  func.func private @mfma_index_A_32x32xf16() -> !index_pair
  func.func private @mfma_index_C_32x32xf32() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  // From futures.mlir
  func.func private @get_global_load_value_vx2(!future_global_read) -> !vx2

  //===--------------------------------------------------------------------===//
  // Tile initialization
  //===--------------------------------------------------------------------===//

  // Initialize a 32x32 f32 accumulator tile to zero.
  func.func private @zero_C_32x32() -> !rt_C_f32 {
    %c0 = arith.constant 0 : i32
    %result = func.call @init_vgprx16(%c0) : (i32) -> !vx16
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory load functions
  //===--------------------------------------------------------------------===//

  // RVW: update to load 32x32 f16 and update all uses
  // Load a 32x8 f16 tile from global memory in MFMA A fragment layout.
  // Lane l loads row (l % 32), cols [(l/32)*4, (l/32)*4 + 4).
  // Each thread loads 4xf16 (8 bytes) via global_load_dwordx2.
  func.func private @load_A_32x32_f16(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !future_global_read {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %elt_size = arith.constant 2 : index
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  func.func private @get_A_32x32_f16(%future: !future_global_read) -> !rt_A_f16 {
    %result = func.call @get_global_load_value_vx2(%future) : (!future_global_read) -> !vx2
    return %result : !rt_A_f16
  }

  // RVW: update to load 32x32 f16 and update all uses
  // Load a 32x8 f16 tile from global memory in MFMA B fragment layout.
  // Uses the same physical layout as A since MFMA computes A @ B^T internally.
  func.func private @load_B_32x32_f16(%ptr: !sx2, %m: index, %n: index, %stride: index) -> !future_global_read {
    %mfma_idx = func.call @mfma_index_A_32x32xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %elt_size = arith.constant 2 : index
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    %value_any = aster_utils.to_any %result : !vx2
    %future = aster_utils.struct_create(%value_any, %tok) : (!aster_utils.any, !amdgcn.read_token<flat>) -> !future_global_read
    return %future : !future_global_read
  }

  func.func private @get_B_32x32_f16(%future: !future_global_read) -> !rt_B_f16 {
    %result = func.call @get_global_load_value_vx2(%future) : (!future_global_read) -> !vx2
    return %result : !rt_B_f16
  }

  //===--------------------------------------------------------------------===//
  // MFMA operation
  //===--------------------------------------------------------------------===//

  // Perform a 32x32x8 matrix multiply-accumulate: D = A @ B^T + C
  // Uses v_mfma_f32_32x32x8_f16 instruction.
  //
  // D[32x32] = A[32x8] @ B[32x8]^T + C[32x32]
  // A and B are f16, C and D are f32.
  func.func private @mfma_f32_32x32x8_f16(%A: !rt_A_f16, %B: !rt_B_f16, %C: !rt_C_f32) -> !rt_C_f32 {
    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_32x32x8_f16> %C, %A, %B, %C
        : !vx2, !vx2, !vx16 -> !vx16
    return %result : !rt_C_f32
  }

  //===--------------------------------------------------------------------===//
  // Global memory store for C tile (32x32 f32)
  //===--------------------------------------------------------------------===//

  // Store a 32x32 f32 C tile to global memory from MFMA C fragment layout.
  // Each thread holds 16xf32 across 16 VGPRs.
  //
  // C fragment layout for 32x32:
  //   col = lane_id % 32
  //   row for register r = row_base + 8*(r/4) + r%4
  //   where row_base = (lane_id / 32) * 4  (0 or 4)
  //
  // We perform 16 separate global_store_dword operations.
  func.func private @store_C_32x32_f32(%tile: !rt_C_f32, %ptr: !sx2, %m: index, %n: index, %stride: index) -> !write_token {
    %mfma_idx = func.call @mfma_index_C_32x32xf32() : () -> !index_pair
    %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"] : !index_pair -> index, index

    %elt_size = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32

    // Split !vx16 into 16 individual VGPRs
    %regs:16 = amdgcn.split_register_range %tile : !vx16

    // Store each of 16 f32 values to global memory.
    // Row for register r: row_base + 8*(r/4) + r%4
    // r=0: rb+0, r=1: rb+1, r=2: rb+2, r=3: rb+3,
    // r=4: rb+8, r=5: rb+9, r=6: rb+10, r=7: rb+11,
    // r=8: rb+16, r=9: rb+17, r=10: rb+18, r=11: rb+19,
    // r=12: rb+24, r=13: rb+25, r=14: rb+26, r=15: rb+27

    %row0 = affine.apply affine_map<()[rb] -> (rb)>()[%row_base]
    %desc0 = aster_utils.struct_create(%m, %n, %row0, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off0 = func.call @tiled_matrix_offset(%desc0) : (!index_descriptor_2level_2d) -> !v
    %tok0 = amdgcn.store global_store_dword data %regs#0 addr %ptr offset d(%off0) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row1 = affine.apply affine_map<()[rb] -> (rb + 1)>()[%row_base]
    %desc1 = aster_utils.struct_create(%m, %n, %row1, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off1 = func.call @tiled_matrix_offset(%desc1) : (!index_descriptor_2level_2d) -> !v
    %tok1 = amdgcn.store global_store_dword data %regs#1 addr %ptr offset d(%off1) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row2 = affine.apply affine_map<()[rb] -> (rb + 2)>()[%row_base]
    %desc2 = aster_utils.struct_create(%m, %n, %row2, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off2 = func.call @tiled_matrix_offset(%desc2) : (!index_descriptor_2level_2d) -> !v
    %tok2 = amdgcn.store global_store_dword data %regs#2 addr %ptr offset d(%off2) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row3 = affine.apply affine_map<()[rb] -> (rb + 3)>()[%row_base]
    %desc3 = aster_utils.struct_create(%m, %n, %row3, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off3 = func.call @tiled_matrix_offset(%desc3) : (!index_descriptor_2level_2d) -> !v
    %tok3 = amdgcn.store global_store_dword data %regs#3 addr %ptr offset d(%off3) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row4 = affine.apply affine_map<()[rb] -> (rb + 8)>()[%row_base]
    %desc4 = aster_utils.struct_create(%m, %n, %row4, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off4 = func.call @tiled_matrix_offset(%desc4) : (!index_descriptor_2level_2d) -> !v
    %tok4 = amdgcn.store global_store_dword data %regs#4 addr %ptr offset d(%off4) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row5 = affine.apply affine_map<()[rb] -> (rb + 9)>()[%row_base]
    %desc5 = aster_utils.struct_create(%m, %n, %row5, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off5 = func.call @tiled_matrix_offset(%desc5) : (!index_descriptor_2level_2d) -> !v
    %tok5 = amdgcn.store global_store_dword data %regs#5 addr %ptr offset d(%off5) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row6 = affine.apply affine_map<()[rb] -> (rb + 10)>()[%row_base]
    %desc6 = aster_utils.struct_create(%m, %n, %row6, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off6 = func.call @tiled_matrix_offset(%desc6) : (!index_descriptor_2level_2d) -> !v
    %tok6 = amdgcn.store global_store_dword data %regs#6 addr %ptr offset d(%off6) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row7 = affine.apply affine_map<()[rb] -> (rb + 11)>()[%row_base]
    %desc7 = aster_utils.struct_create(%m, %n, %row7, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off7 = func.call @tiled_matrix_offset(%desc7) : (!index_descriptor_2level_2d) -> !v
    %tok7 = amdgcn.store global_store_dword data %regs#7 addr %ptr offset d(%off7) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row8 = affine.apply affine_map<()[rb] -> (rb + 16)>()[%row_base]
    %desc8 = aster_utils.struct_create(%m, %n, %row8, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off8 = func.call @tiled_matrix_offset(%desc8) : (!index_descriptor_2level_2d) -> !v
    %tok8 = amdgcn.store global_store_dword data %regs#8 addr %ptr offset d(%off8) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row9 = affine.apply affine_map<()[rb] -> (rb + 17)>()[%row_base]
    %desc9 = aster_utils.struct_create(%m, %n, %row9, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off9 = func.call @tiled_matrix_offset(%desc9) : (!index_descriptor_2level_2d) -> !v
    %tok9 = amdgcn.store global_store_dword data %regs#9 addr %ptr offset d(%off9) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row10 = affine.apply affine_map<()[rb] -> (rb + 18)>()[%row_base]
    %desc10 = aster_utils.struct_create(%m, %n, %row10, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off10 = func.call @tiled_matrix_offset(%desc10) : (!index_descriptor_2level_2d) -> !v
    %tok10 = amdgcn.store global_store_dword data %regs#10 addr %ptr offset d(%off10) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row11 = affine.apply affine_map<()[rb] -> (rb + 19)>()[%row_base]
    %desc11 = aster_utils.struct_create(%m, %n, %row11, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off11 = func.call @tiled_matrix_offset(%desc11) : (!index_descriptor_2level_2d) -> !v
    %tok11 = amdgcn.store global_store_dword data %regs#11 addr %ptr offset d(%off11) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row12 = affine.apply affine_map<()[rb] -> (rb + 24)>()[%row_base]
    %desc12 = aster_utils.struct_create(%m, %n, %row12, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off12 = func.call @tiled_matrix_offset(%desc12) : (!index_descriptor_2level_2d) -> !v
    %tok12 = amdgcn.store global_store_dword data %regs#12 addr %ptr offset d(%off12) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row13 = affine.apply affine_map<()[rb] -> (rb + 25)>()[%row_base]
    %desc13 = aster_utils.struct_create(%m, %n, %row13, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off13 = func.call @tiled_matrix_offset(%desc13) : (!index_descriptor_2level_2d) -> !v
    %tok13 = amdgcn.store global_store_dword data %regs#13 addr %ptr offset d(%off13) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row14 = affine.apply affine_map<()[rb] -> (rb + 26)>()[%row_base]
    %desc14 = aster_utils.struct_create(%m, %n, %row14, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off14 = func.call @tiled_matrix_offset(%desc14) : (!index_descriptor_2level_2d) -> !v
    %tok14 = amdgcn.store global_store_dword data %regs#14 addr %ptr offset d(%off14) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    %row15 = affine.apply affine_map<()[rb] -> (rb + 27)>()[%row_base]
    %desc15 = aster_utils.struct_create(%m, %n, %row15, %col, %stride, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off15 = func.call @tiled_matrix_offset(%desc15) : (!index_descriptor_2level_2d) -> !v
    %tok15 = amdgcn.store global_store_dword data %regs#15 addr %ptr offset d(%off15) + c(%c0_i32) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    return %tok15 : !write_token
  }
}
