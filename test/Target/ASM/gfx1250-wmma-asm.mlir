// RUN: aster-opt %s --aster-to-amdgcn | aster-translate --mlir-to-asm | FileCheck %s

// CHECK-LABEL: wmma_ops:
//       CHECK: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
//  CHECK-NEXT: v_wmma_f32_16x16x32_bf16 v[24:31], v[0:7], v[8:15], v[24:31]
//  CHECK-NEXT: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] matrix_a_reuse matrix_b_reuse
//  CHECK-NEXT: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] neg_hi:[0,0,1]
//       CHECK: s_endpgm

amdgcn.module @wmma_asm_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @wmma_ops attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %a = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %b = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %df = lsir.alloca : !amdgcn.vgpr<[16 : 24]>
    %db = lsir.alloca : !amdgcn.vgpr<[24 : 32]>

    amdgcn.v_wmma_f32_16x16x32_f16 outs(%df) ins(%a, %b, %df)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.v_wmma_f32_16x16x32_bf16 outs(%db) ins(%a, %b, %db)
        : outs(!amdgcn.vgpr<[24 : 32]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[24 : 32]>)

    // Optional modifiers (matrix_a_reuse / matrix_b_reuse, then neg_lo / neg_hi on C).
    amdgcn.v_wmma_f32_16x16x32_f16 outs(%df) ins(%a, %b, %df)
        matrix_a_reuse(unit) matrix_b_reuse(unit)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.v_wmma_f32_16x16x32_f16 outs(%df) ins(%a, %b, %df)
        neg_lo(array<i1: false, false, true>) neg_hi(array<i1: false, false, true>)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)

    amdgcn.end_kernel
  }
}
