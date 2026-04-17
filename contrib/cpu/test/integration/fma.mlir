x86.module @avx_fma target_isa = #x86.isa<avx> {
  func.func @test_avx_fma(
      %a: !x86.gpr<rdi>,
      %b: !x86.gpr<rsi>,
      %c: !x86.gpr<rdx>) {
    %va = x86.avx.load vmovups %a : !x86.gpr<rdi> -> !x86.avx.xmm<0>
    %vb = x86.avx.load vmovups %b : !x86.gpr<rsi> -> !x86.avx.xmm<1>
    %vc = x86.avx.load vmovups %c : !x86.gpr<rdx> -> !x86.avx.xmm<2>
    %vr = x86.avx.vfmadd231ps %vc, %va, %vb
        : (!x86.avx.xmm<2>, !x86.avx.xmm<0>, !x86.avx.xmm<1>)
            -> !x86.avx.xmm<2>
    x86.avx.store vmovups %c, %vr : !x86.gpr<rdx>, !x86.avx.xmm<2>
    return
  }
}

x86.module @avx2_fma target_isa = #x86.isa<avx2> {
  func.func @test_avx2_fma(
      %a: !x86.gpr<rdi>,
      %b: !x86.gpr<rsi>,
      %c: !x86.gpr<rdx>) {
    %va = x86.avx2.load vmovups %a : !x86.gpr<rdi> -> !x86.avx2.ymm<0>
    %vb = x86.avx2.load vmovups %b : !x86.gpr<rsi> -> !x86.avx2.ymm<1>
    %vc = x86.avx2.load vmovups %c : !x86.gpr<rdx> -> !x86.avx2.ymm<2>
    %vr = x86.avx2.vfmadd231ps %vc, %va, %vb
        : (!x86.avx2.ymm<2>, !x86.avx2.ymm<0>, !x86.avx2.ymm<1>)
            -> !x86.avx2.ymm<2>
    x86.avx2.store vmovups %c, %vr : !x86.gpr<rdx>, !x86.avx2.ymm<2>
    return
  }
}

x86.module @avx512_fma target_isa = #x86.isa<avx512> {
  func.func @test_avx512_fma(
      %a: !x86.gpr<rdi>,
      %b: !x86.gpr<rsi>,
      %c: !x86.gpr<rdx>) {
    %va = x86.avx512.load vmovups %a : !x86.gpr<rdi> -> !x86.avx512.zmm<0>
    %vb = x86.avx512.load vmovups %b : !x86.gpr<rsi> -> !x86.avx512.zmm<1>
    %vc = x86.avx512.load vmovups %c : !x86.gpr<rdx> -> !x86.avx512.zmm<2>
    %vr = x86.avx512.vfmadd231ps %vc, %va, %vb
        : (!x86.avx512.zmm<2>, !x86.avx512.zmm<0>, !x86.avx512.zmm<1>)
            -> !x86.avx512.zmm<2>
    x86.avx512.store vmovups %c, %vr : !x86.gpr<rdx>, !x86.avx512.zmm<2>
    return
  }
}
