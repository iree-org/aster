// RUN: aster-cpu-translate --mlir-to-x86-asm %s \
// RUN:   | %llvm_mc -triple=x86_64-unknown-linux-gnu \
// RUN:       -mattr=+amx-tile,+amx-bf16 -filetype=obj -o %t.o
// RUN: %llvm_objdump -d -t --mattr=+amx-tile,+amx-bf16 %t.o | FileCheck %s

// CHECK:       SYMBOL TABLE:
// CHECK:       {{.*}} g {{.*}}F .text {{.*}} test_tdpbf16ps

// CHECK-LABEL: <test_tdpbf16ps>:
// CHECK:       {{[0-9a-f ]+}}ldtilecfg{{[ \t]+}}{{.*}}(%rip)
// CHECK:       {{[0-9a-f ]+}}tileloadd{{[ \t]+}}(%rdi,%rcx), %tmm1
// CHECK:       {{[0-9a-f ]+}}tileloadd{{[ \t]+}}(%rsi,%rcx), %tmm2
// CHECK:       {{[0-9a-f ]+}}tilezero{{[ \t]+}}%tmm0
// CHECK:       {{[0-9a-f ]+}}tdpbf16ps{{[ \t]+}}%tmm2, %tmm1, %tmm0
// CHECK:       {{[0-9a-f ]+}}tilestored{{[ \t]+}}%tmm0, (%rdx,%rcx)
// CHECK:       {{[0-9a-f ]+}}tilerelease
// CHECK:       {{[0-9a-f ]+}}retq
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
