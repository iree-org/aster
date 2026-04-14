// RUN: aster-cpu-translate --mlir-to-amx-asm %s | FileCheck %s

// CHECK:      .section .rodata
// CHECK:      .Lcfg_test_tdpbf16ps:
// CHECK:       .byte 1
// CHECK:       .text
// CHECK:       .globl test_tdpbf16ps
// CHECK:       .p2align 4
// CHECK:       .type test_tdpbf16ps,@function
// CHECK:      test_tdpbf16ps:
// CHECK:       ldtilecfg .Lcfg_test_tdpbf16ps(%rip)
// CHECK:       tileloadd (%rdi, %rcx, 1), %tmm1
// CHECK:       tileloadd (%rsi, %rcx, 1), %tmm2
// CHECK:       tileloadd (%rdx, %rcx, 1), %tmm0
// CHECK:       tdpbf16ps %tmm2, %tmm1, %tmm0
// CHECK:       tilestored %tmm0, (%rdx, %rcx, 1)
// CHECK:       tilerelease
// CHECK:       retq
// CHECK:      .Lfunc_end0:
// CHECK:       .size test_tdpbf16ps, .Lfunc_end0-test_tdpbf16ps
func.func @test_tdpbf16ps(
    %a: !x86.gpr<rdi>,
    %b: !x86.gpr<rsi>,
    %c: !x86.gpr<rdx>,
    %stride: !x86.gpr<rcx>) {
  amx.ldtilecfg
  %ta = amx.tileloadd %a, %stride : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !amx.tile<"tmm1", 16 x 32 x bf16>
  %tb = amx.tileloadd %b, %stride : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !amx.tile<"tmm2", 16 x 32 x bf16>
  %tc = amx.tileloadd %c, %stride : (!x86.gpr<rdx>, !x86.gpr<rcx>) -> !amx.tile<"tmm0", 16 x 16 x f32>
  %td = amx.tdpbf16ps %tc, %ta, %tb
      : (!amx.tile<"tmm0", 16 x 16 x f32>, !amx.tile<"tmm1", 16 x 32 x bf16>, !amx.tile<"tmm2", 16 x 32 x bf16>) -> !amx.tile<"tmm0", 16 x 16 x f32>
  amx.tilestored %c, %stride, %td : !x86.gpr<rdx>, !x86.gpr<rcx>, !amx.tile<"tmm0", 16 x 16 x f32>
  amx.tilerelease
  return
}
