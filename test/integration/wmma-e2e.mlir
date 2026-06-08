// RUN: aster-opt %s --verify-roundtrip

// gfx1250/gfx1251 WMMA forms exercised through the e2e (compile + HSACO assembly,
// execution skipped -- no silicon): bare f16/bf16, the matrix_a_reuse /
// matrix_b_reuse hints, and neg_lo / neg_hi on the C operand.
amdgcn.module @wmma_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @wmma_f16 attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %a = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %b = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %d = lsir.alloca : !amdgcn.vgpr<[16 : 24]>
    amdgcn.v_wmma_f32_16x16x32_f16 outs(%d) ins(%a, %b, %d)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @wmma_bf16 attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %a = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %b = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %d = lsir.alloca : !amdgcn.vgpr<[16 : 24]>
    amdgcn.v_wmma_f32_16x16x32_bf16 outs(%d) ins(%a, %b, %d)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @wmma_f16_reuse attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %a = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %b = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %d = lsir.alloca : !amdgcn.vgpr<[16 : 24]>
    amdgcn.v_wmma_f32_16x16x32_f16 outs(%d) ins(%a, %b, %d)
        matrix_a_reuse(unit) matrix_b_reuse(unit)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.end_kernel
  }

  amdgcn.kernel @wmma_f16_neg attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %a = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %b = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %d = lsir.alloca : !amdgcn.vgpr<[16 : 24]>
    amdgcn.v_wmma_f32_16x16x32_f16 outs(%d) ins(%a, %b, %d)
        neg_lo(array<i1: false, false, true>) neg_hi(array<i1: false, false, true>)
        : outs(!amdgcn.vgpr<[16 : 24]>) ins(!amdgcn.vgpr<[0 : 8]>, !amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>)
    amdgcn.end_kernel
  }
}
