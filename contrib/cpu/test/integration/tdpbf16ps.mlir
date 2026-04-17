// TODO: is func.func acceptable here, do we need a special kernel op, what
// about ABI?
func.func @test_tdpbf16ps(
    %a: !x86.gpr<rdi>,
    %b: !x86.gpr<rsi>,
    %c: !x86.gpr<rdx>,
    %stride: !x86.gpr<rcx>) {
  x86.amx.ldtilecfg
  %ta = x86.amx.tileloadd %a, %stride : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !x86.amx.tmm<1, 16 x 32 x bf16>
  %tb = x86.amx.tileloadd %b, %stride : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !x86.amx.tmm<2, 16 x 32 x bf16>
  %tc = x86.amx.tilezero : !x86.amx.tmm<0, 16 x 16 x f32>
  %td = x86.amx.tdpbf16ps %tc, %ta, %tb
      : (!x86.amx.tmm<0, 16 x 16 x f32>, !x86.amx.tmm<1, 16 x 32 x bf16>, !x86.amx.tmm<2, 16 x 32 x bf16>) -> !x86.amx.tmm<0, 16 x 16 x f32>
  x86.amx.tilestored %c, %stride, %td : !x86.gpr<rdx>, !x86.gpr<rcx>, !x86.amx.tmm<0, 16 x 16 x f32>
  x86.amx.tilerelease
  return
}
