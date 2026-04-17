// RUN: aster-cpu-translate --mlir-to-amx-asm %s | FileCheck %s

// CHECK-LABEL: test_avx_fma:
// CHECK:       vmovaps (%rdi), %xmm0
// CHECK:       vmovaps (%rsi), %xmm1
// CHECK:       vmovaps (%rdx), %xmm2
// CHECK:       vfmadd231ps %xmm1, %xmm0, %xmm2
// CHECK:       vmovaps %xmm2, (%rdx)
// CHECK:       retq

// CHECK-LABEL: test_avx2_fma:
// CHECK:       vmovaps (%rdi), %ymm0
// CHECK:       vmovaps (%rsi), %ymm1
// CHECK:       vmovaps (%rdx), %ymm2
// CHECK:       vfmadd231ps %ymm1, %ymm0, %ymm2
// CHECK:       vmovaps %ymm2, (%rdx)
// CHECK:       retq

// CHECK-LABEL: test_avx512_fma:
// CHECK:       vmovaps (%rdi), %zmm0
// CHECK:       vmovaps (%rsi), %zmm1
// CHECK:       vmovaps (%rdx), %zmm2
// CHECK:       vfmadd231ps %zmm1, %zmm0, %zmm2
// CHECK:       vmovaps %zmm2, (%rdx)
// CHECK:       retq

x86.module @avx_fma target_isa = #x86.isa<avx> {
  func.func @test_avx_fma(
      %a: !x86.gpr<rdi>, %b: !x86.gpr<rsi>, %c: !x86.gpr<rdx>) {
    %va = x86.avx.load vmovaps %a : !x86.gpr<rdi> -> !x86.avx.xmm<0>
    %vb = x86.avx.load vmovaps %b : !x86.gpr<rsi> -> !x86.avx.xmm<1>
    %vc = x86.avx.load vmovaps %c : !x86.gpr<rdx> -> !x86.avx.xmm<2>
    %vr = x86.avx.vfmadd231ps %vc, %va, %vb
        : (!x86.avx.xmm<2>, !x86.avx.xmm<0>, !x86.avx.xmm<1>)
            -> !x86.avx.xmm<2>
    x86.avx.store vmovaps %c, %vr : !x86.gpr<rdx>, !x86.avx.xmm<2>
    return
  }
}

x86.module @avx2_fma target_isa = #x86.isa<avx2> {
  func.func @test_avx2_fma(
      %a: !x86.gpr<rdi>, %b: !x86.gpr<rsi>, %c: !x86.gpr<rdx>) {
    %va = x86.avx2.load vmovaps %a : !x86.gpr<rdi> -> !x86.avx2.ymm<0>
    %vb = x86.avx2.load vmovaps %b : !x86.gpr<rsi> -> !x86.avx2.ymm<1>
    %vc = x86.avx2.load vmovaps %c : !x86.gpr<rdx> -> !x86.avx2.ymm<2>
    %vr = x86.avx2.vfmadd231ps %vc, %va, %vb
        : (!x86.avx2.ymm<2>, !x86.avx2.ymm<0>, !x86.avx2.ymm<1>)
            -> !x86.avx2.ymm<2>
    x86.avx2.store vmovaps %c, %vr : !x86.gpr<rdx>, !x86.avx2.ymm<2>
    return
  }
}

x86.module @avx512_fma target_isa = #x86.isa<avx512> {
  func.func @test_avx512_fma(
      %a: !x86.gpr<rdi>, %b: !x86.gpr<rsi>, %c: !x86.gpr<rdx>) {
    %va = x86.avx512.load vmovaps %a : !x86.gpr<rdi> -> !x86.avx512.zmm<0>
    %vb = x86.avx512.load vmovaps %b : !x86.gpr<rsi> -> !x86.avx512.zmm<1>
    %vc = x86.avx512.load vmovaps %c : !x86.gpr<rdx> -> !x86.avx512.zmm<2>
    %vr = x86.avx512.vfmadd231ps %vc, %va, %vb
        : (!x86.avx512.zmm<2>, !x86.avx512.zmm<0>, !x86.avx512.zmm<1>)
            -> !x86.avx512.zmm<2>
    x86.avx512.store vmovaps %c, %vr : !x86.gpr<rdx>, !x86.avx512.zmm<2>
    return
  }
}
