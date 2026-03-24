// RUN: mlir-air-opt %s \
// RUN:   --transform-interpreter --canonicalize \
// RUN:   --convert-linalg-to-amdgcn \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir,%p/../../../mlir_kernels/library/common/indexing.mlir,%p/../../../mlir_kernels/library/common/indexing_ptr.mlir,%p/../../../mlir_kernels/library/common/futures.mlir,%p/../../../contrib/kittens/library/compute_16x16_f16.mlir,%p/../../../contrib/kittens/library/global_16x64_b.mlir,%p/../../../contrib/kittens/library/lds_16x64_b.mlir,%p/../../../contrib/kittens/library/lds_mfma_16x64_b.mlir" \
// RUN:   --inline --symbol-dce --canonicalize \
// RUN:   --mlir-air-to-asm \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// CHECK-LABEL: matmul_f16_32x32:
// CHECK:   global_load_dwordx4
// CHECK:   ds_write_b64
// CHECK:   ds_read_b64
// CHECK:   v_mfma_f32_16x16x16_f16
// CHECK:   global_store_dword
// CHECK:   s_endpgm

!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>


module attributes {transform.with_named_sequence} {
  amdgcn.library @linalg_lib isa = [#amdgcn.isa<cdna3>] {
    // Kittens compute (compute_16x16_f16.mlir)
    func.func private @zero_C() -> !ax4
    func.func private @mfma_f32_16x16x16_f16(!vx2, !vx2, !ax4) -> !ax4
    func.func private @store_global_C_mfma_f32_16x16x16_f16(
        !ax4, !aster_utils.any, index, index, index)

    // Kittens global loads (global_16x64_b.mlir)
    func.func private @prepare_ptr(!sx2) -> !aster_utils.any
    func.func private @load_global_tile_16x64_b(
        !aster_utils.any, index, index, index) -> !future_global_read

    // Kittens LDS ops (lds_16x64_b.mlir)
    func.func private @store_global_tile_to_lds_16x64_b(
        index, !future_global_read) -> (!lds_write_token, !lds_write_token)

    // Kittens MFMA-aware LDS reads (lds_mfma_16x64_b.mlir)
    func.func private @load_lds_A_swizzled(
        index, index, index) -> !future_lds_read
    func.func private @load_lds_B_swizzled(
        index, index, index) -> !future_lds_read

    // Futures (futures.mlir)
    func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

    // Copy global -> LDS via kittens: prepare_ptr + tile load + LDS write.
    // Args decomposed by C++ pass: (!sx2 ptr, index byte_stride, index lds_dst).
    func.func private @copy_f16_16x32(
        %src_ptr: !sx2, %src_stride: index, %lds_dst: index) {
      %ptr = func.call @prepare_ptr(%src_ptr) : (!sx2) -> !aster_utils.any
      %c0 = arith.constant 0 : index
      %gfut = func.call @load_global_tile_16x64_b(%ptr, %c0, %c0, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t0, %t1 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst, %gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %t0 : !lds_write_token
      amdgcn.wait deps %t1 : !lds_write_token
      return
    }

    // MFMA matmul via kittens: A,B from LDS, C store to global.
    // Verbatim kittens pattern: zero_C, 2x (load_lds + MFMA), store_global_C.
    // Args decomposed by C++ pass: (index lds_A, index lds_B, !sx2 C_ptr, index C_stride).
    func.func private @mfma_matmul_f16_16x32(
        %lds_A: index, %lds_B: index, %C_ptr: !sx2, %C_stride: index) {
      %C_prepared = func.call @prepare_ptr(%C_ptr) : (!sx2) -> !aster_utils.any

      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %acc = func.call @zero_C() : () -> !ax4

      // K0 sub-tile (byte offset 0)
      %A0f = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A0 = func.call @get_lds_read_value_vx2(%A0f) : (!future_lds_read) -> !vx2
      %B0f = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B0 = func.call @get_lds_read_value_vx2(%B0f) : (!future_lds_read) -> !vx2
      %acc0 = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %acc)
          : (!vx2, !vx2, !ax4) -> !ax4

      // K1 sub-tile (byte offset 32)
      %A1f = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A1 = func.call @get_lds_read_value_vx2(%A1f) : (!future_lds_read) -> !vx2
      %B1f = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B1 = func.call @get_lds_read_value_vx2(%B1f) : (!future_lds_read) -> !vx2
      %acc1 = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %acc0)
          : (!vx2, !vx2, !ax4) -> !ax4

      // Store C via kittens (fire-and-forget)
      func.call @store_global_C_mfma_f32_16x16x16_f16(
          %acc1, %C_prepared, %c0, %c0, %C_stride)
          : (!ax4, !aster_utils.any, index, index, index) -> ()
      return
    }

    func.func private @fill_f16_16x32(%val: f16, %lds_dst: index) { return }
  }

  amdgcn.module @matmul_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
    func.func @matmul_f16_32x32(
        %A: memref<32x32xf16, #amdgcn.addr_space<global, read_write>>,
        %B: memref<32x32xf16, #amdgcn.addr_space<global, read_write>>,
        %C: memref<32x32xf32, #amdgcn.addr_space<global, read_write>>)
        attributes {gpu.kernel} {
      // matmul_transpose_b: C[m,n] += A[m,k] * B[n,k]
      // B is stored as NxK (transposed), matching kittens' 16x64_b tile layout.
      linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : memref<32x32xf16, #amdgcn.addr_space<global, read_write>>,
                    memref<32x32xf16, #amdgcn.addr_space<global, read_write>>)
        outs(%C : memref<32x32xf32, #amdgcn.addr_space<global, read_write>>) {
      ^bb0(%a: f16, %b: f16, %c: f32):
        %a_ext = arith.extf %a : f16 to f32
        %b_ext = arith.extf %b : f16 to f32
        %prod = arith.mulf %a_ext, %b_ext : f32
        %sum = arith.addf %c, %prod : f32
        linalg.yield %sum : f32
      }
      return
    }
  }

  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %t, %lm, %ln, %lk = transform.structured.tile_using_for %matmul
        tile_sizes [16, 16, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                   !transform.any_op, !transform.any_op)
    %promoted = transform.structured.promote %t {
      operands_to_promote = [0, 1],
      memory_space = #gpu.address_space<workgroup>,
      use_full_tiles_by_default,
      use_alloca
    } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
