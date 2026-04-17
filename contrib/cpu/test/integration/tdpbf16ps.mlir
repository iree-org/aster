func.func @test_tdpbf16ps(
    %a: !x86.gpr<rdi>,
    %b: !x86.gpr<rsi>,
    %c: !x86.gpr<rdx>,
    %stride: !x86.gpr<rcx>) {
  amx.ldtilecfg
  %ta = amx.tileloadd %a, %stride : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !amx.tile<tmm1, 16 x 32 x bf16>
  %tb = amx.tileloadd %b, %stride : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !amx.tile<tmm2, 16 x 32 x bf16>
  %tc = amx.tilezero : !amx.tile<tmm0, 16 x 16 x f32>
  %td = amx.tdpbf16ps %tc, %ta, %tb
      : (!amx.tile<tmm0, 16 x 16 x f32>, !amx.tile<tmm1, 16 x 32 x bf16>, !amx.tile<tmm2, 16 x 32 x bf16>) -> !amx.tile<tmm0, 16 x 16 x f32>
  amx.tilestored %c, %stride, %td : !x86.gpr<rdx>, !x86.gpr<rcx>, !amx.tile<tmm0, 16 x 16 x f32>
  amx.tilerelease
  return
}
