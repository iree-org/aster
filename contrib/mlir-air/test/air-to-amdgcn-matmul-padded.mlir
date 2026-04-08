// Padded matmul: actual M=40, N=40, K=64.
// Device-side padding via air-split-launch-for-padding.
//
// The kernel has air.launch (3x3 grid) written directly — no transform,
// no air-wrap-func-with-parallel. Each block computes one 16x16 output tile.
//
// A, B: actual 40x64, C: 48x48 (padded for safe boundary stores).
// Boundary tiles (last row/col) copy fewer rows from global, with LDS
// zero-filled first. split-launch adds pad_after on boundary DMAs.
// ConvertToAMDGCNLibraryCalls emits fill + copy_padded for those.
//
// Host extracts valid C[0:40, 0:40] after kernel.

!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

module {
  amdgcn.library @linalg_lib isa = [#amdgcn.isa<cdna3>] {
    func.func private @zero_C() -> !ax4
    func.func private @mfma_f32_16x16x16_f16(!vx2, !vx2, !ax4) -> !ax4
    func.func private @store_global_C_mfma_f32_16x16x16_f16(
        !ax4, !aster_utils.any, index, index, index)
    func.func private @prepare_ptr(!sx2) -> !aster_utils.any
    func.func private @load_global_tile_16x64_b(
        !aster_utils.any, index, index, index) -> !future_global_read
    func.func private @store_global_tile_to_lds_16x64_b(
        index, !future_global_read) -> (!lds_write_token, !lds_write_token)
    func.func private @load_lds_A_swizzled(
        index, index, index) -> !future_lds_read
    func.func private @load_lds_B_swizzled(
        index, index, index) -> !future_lds_read
    func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2
    func.func private @thread_tile_pos_16x64_b() -> (index, index)
    func.func private @tiled_row_byte_off(index, index, index, index, index, index) -> index
    func.func private @load_global_at_byte_off(!aster_utils.any, index) -> !future_global_read
    func.func private @fill_lds_16x64_b(index)

    func.func private @copy_f16_16x64(
        %src_ptr: !sx2, %src_stride: index,
        %row_offset: index, %col_offset: index,
        %lds_dst: index) {
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %col1 = arith.addi %col_offset, %c32 : index
      %lds_dst1 = arith.addi %lds_dst, %c1024 : index
      %ptr = func.call @prepare_ptr(%src_ptr) : (!sx2) -> !aster_utils.any
      %gfut0 = func.call @load_global_tile_16x64_b(
          %ptr, %row_offset, %col_offset, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t0, %t1 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst, %gfut0)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %gfut1 = func.call @load_global_tile_16x64_b(
          %ptr, %row_offset, %col1, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t2, %t3 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst1, %gfut1)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %t0 : !lds_write_token
      amdgcn.wait deps %t1 : !lds_write_token
      amdgcn.wait deps %t2 : !lds_write_token
      amdgcn.wait deps %t3 : !lds_write_token
      return
    }

    func.func private @copy_f16_16x64_padded(
        %src_ptr: !sx2, %src_stride: index,
        %row_offset: index, %col_offset: index,
        %actual_rows: index,
        %lds_dst: index) {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %ptr = func.call @prepare_ptr(%src_ptr) : (!sx2) -> !aster_utils.any
      %my_row, %my_col_byte = func.call @thread_tile_pos_16x64_b()
          : () -> (index, index)
      %max_row = arith.subi %actual_rows, %c1 : index
      %clamped_row = arith.minui %my_row, %max_row : index
      %byte_off0 = func.call @tiled_row_byte_off(
          %row_offset, %clamped_row, %col_offset, %my_col_byte, %src_stride, %c2)
          : (index, index, index, index, index, index) -> index
      %gfut0 = func.call @load_global_at_byte_off(%ptr, %byte_off0)
          : (!aster_utils.any, index) -> !future_global_read
      %t0, %t1 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst, %gfut0)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %col1 = arith.addi %col_offset, %c32 : index
      %lds_dst1 = arith.addi %lds_dst, %c1024 : index
      %byte_off1 = func.call @tiled_row_byte_off(
          %row_offset, %clamped_row, %col1, %my_col_byte, %src_stride, %c2)
          : (index, index, index, index, index, index) -> index
      %gfut1 = func.call @load_global_at_byte_off(%ptr, %byte_off1)
          : (!aster_utils.any, index) -> !future_global_read
      %t2, %t3 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst1, %gfut1)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %t0 : !lds_write_token
      amdgcn.wait deps %t1 : !lds_write_token
      amdgcn.wait deps %t2 : !lds_write_token
      amdgcn.wait deps %t3 : !lds_write_token
      return
    }

    func.func private @mfma_matmul_f16_16x64(
        %lds_A: index, %lds_B: index,
        %C_ptr: !sx2, %C_stride: index,
        %C_row_offset: index, %C_col_offset: index) {
      %C_prepared = func.call @prepare_ptr(%C_ptr) : (!sx2) -> !aster_utils.any
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %lds_A2 = arith.addi %lds_A, %c1024 : index
      %lds_B2 = arith.addi %lds_B, %c1024 : index
      %acc = func.call @zero_C() : () -> !ax4
      %A0f = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A0 = func.call @get_lds_read_value_vx2(%A0f) : (!future_lds_read) -> !vx2
      %B0f = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B0 = func.call @get_lds_read_value_vx2(%B0f) : (!future_lds_read) -> !vx2
      %acc0 = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %acc)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A1f = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A1 = func.call @get_lds_read_value_vx2(%A1f) : (!future_lds_read) -> !vx2
      %B1f = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B1 = func.call @get_lds_read_value_vx2(%B1f) : (!future_lds_read) -> !vx2
      %acc1 = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %acc0)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A2f = func.call @load_lds_A_swizzled(%lds_A2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A2 = func.call @get_lds_read_value_vx2(%A2f) : (!future_lds_read) -> !vx2
      %B2f = func.call @load_lds_B_swizzled(%lds_B2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B2 = func.call @get_lds_read_value_vx2(%B2f) : (!future_lds_read) -> !vx2
      %acc2 = func.call @mfma_f32_16x16x16_f16(%A2, %B2, %acc1)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A3f = func.call @load_lds_A_swizzled(%lds_A2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A3 = func.call @get_lds_read_value_vx2(%A3f) : (!future_lds_read) -> !vx2
      %B3f = func.call @load_lds_B_swizzled(%lds_B2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B3 = func.call @get_lds_read_value_vx2(%B3f) : (!future_lds_read) -> !vx2
      %acc3 = func.call @mfma_f32_16x16x16_f16(%A3, %B3, %acc2)
          : (!vx2, !vx2, !ax4) -> !ax4
      func.call @store_global_C_mfma_f32_16x16x16_f16(
          %acc3, %C_prepared, %C_row_offset, %C_col_offset, %C_stride)
          : (!ax4, !aster_utils.any, index, index, index) -> ()
      return
    }

    func.func private @fill_f16_16x64(%val: f16, %lds_dst: index) {
      func.call @fill_lds_16x64_b(%lds_dst) : (index) -> ()
      return
    }
  }

  amdgcn.module @matmul_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
    func.func @matmul_f16_40x40(
        %A: memref<40x64xf16>, %B: memref<40x64xf16>, %C: memref<48x48xf32>)
        attributes {gpu.kernel} {
      %c3 = arith.constant 3 : index

      air.launch (%m_id, %n_id) in (%m_sz=%c3, %n_sz=%c3)
          args(%a=%A, %b=%B, %c=%C)
          : memref<40x64xf16>, memref<40x64xf16>, memref<48x48xf32>
          attributes {air.actual_sizes = array<i64: 40, 40, 1>} {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c40 = arith.constant 40 : index
        %cst_f16 = arith.constant 0.000000e+00 : f16

        // Tile offsets from launch block indices.
        %m_off = arith.muli %m_id, %c16 : index
        %n_off = arith.muli %n_id, %c16 : index

        // A tile: copy min(16, 40-m_off) rows from global to LDS.
        %m_rem = arith.subi %c40, %m_off : index
        %m_size = arith.minui %c16, %m_rem : index
        %lds_a = memref.alloc() : memref<16x64xf16, 2>
        linalg.fill ins(%cst_f16 : f16) outs(%lds_a : memref<16x64xf16, 2>)
        %a_sub = memref.subview %a[%m_off, 0] [%m_size, 64] [1, 1]
            : memref<40x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
        %lds_a_sub = memref.subview %lds_a[0, 0] [%m_size, 64] [1, 1]
            : memref<16x64xf16, 2> to memref<?x64xf16, strided<[64, 1]>, 2>
        memref.copy %a_sub, %lds_a_sub
            : memref<?x64xf16, strided<[64, 1], offset: ?>>
              to memref<?x64xf16, strided<[64, 1]>, 2>

        // B tile: copy min(16, 40-n_off) rows from global to LDS.
        %n_rem = arith.subi %c40, %n_off : index
        %n_size = arith.minui %c16, %n_rem : index
        %lds_b = memref.alloc() : memref<16x64xf16, 2>
        linalg.fill ins(%cst_f16 : f16) outs(%lds_b : memref<16x64xf16, 2>)
        %b_sub = memref.subview %b[%n_off, 0] [%n_size, 64] [1, 1]
            : memref<40x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
        %lds_b_sub = memref.subview %lds_b[0, 0] [%n_size, 64] [1, 1]
            : memref<16x64xf16, 2> to memref<?x64xf16, strided<[64, 1]>, 2>
        memref.copy %b_sub, %lds_b_sub
            : memref<?x64xf16, strided<[64, 1], offset: ?>>
              to memref<?x64xf16, strided<[64, 1]>, 2>

        // Matmul on full 16x64 LDS tiles → 16x16 output written to C.
        // C is 48x48 so writing 16x16 at any (m_off, n_off) is safe.
        %c_sub = memref.subview %c[%m_off, %n_off] [16, 16] [1, 1]
            : memref<48x48xf32> to memref<16x16xf32, strided<[48, 1], offset: ?>>
        linalg.generic {
          indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (n, k)>,
            affine_map<(m, n, k) -> (m, n)>
          ],
          iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%lds_a, %lds_b : memref<16x64xf16, 2>, memref<16x64xf16, 2>)
          outs(%c_sub : memref<16x16xf32, strided<[48, 1], offset: ?>>) {
        ^bb0(%av: f16, %bv: f16, %cv: f32):
          %a_ext = arith.extf %av : f16 to f32
          %b_ext = arith.extf %bv : f16 to f32
          %prod = arith.mulf %a_ext, %b_ext : f32
          %sum = arith.addf %cv, %prod : f32
          linalg.yield %sum : f32
        }
      }
      return
    }
  }
}
