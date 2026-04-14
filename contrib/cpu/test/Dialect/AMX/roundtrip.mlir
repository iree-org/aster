// RUN: aster-cpu-opt %s | aster-cpu-opt | FileCheck %s

// CHECK-LABEL: func.func @amx_roundtrip
func.func @amx_roundtrip(
    %a: !x86.gpr<rdi>,
    %b: !x86.gpr<rsi>,
    %c: !x86.gpr<rdx>,
    %stride: !x86.gpr<rcx>) {
  // CHECK: amx.ldtilecfg
  amx.ldtilecfg

  // CHECK: amx.tileloadd %{{.*}}, %{{.*}} : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !amx.tile<"tmm1", 16 x 32 x bf16>
  %ta = amx.tileloadd %a, %stride : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !amx.tile<"tmm1", 16 x 32 x bf16>

  // CHECK: amx.tileloadd %{{.*}}, %{{.*}} : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !amx.tile<"tmm2", 16 x 32 x bf16>
  %tb = amx.tileloadd %b, %stride : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !amx.tile<"tmm2", 16 x 32 x bf16>

  // CHECK: amx.tileloadd %{{.*}}, %{{.*}} : (!x86.gpr<rdx>, !x86.gpr<rcx>) -> !amx.tile<"tmm0", 16 x 16 x f32>
  %tc = amx.tileloadd %c, %stride : (!x86.gpr<rdx>, !x86.gpr<rcx>) -> !amx.tile<"tmm0", 16 x 16 x f32>

  // CHECK: amx.tdpbf16ps %{{.*}}, %{{.*}}, %{{.*}}
  %td = amx.tdpbf16ps %tc, %ta, %tb
      : (!amx.tile<"tmm0", 16 x 16 x f32>, !amx.tile<"tmm1", 16 x 32 x bf16>, !amx.tile<"tmm2", 16 x 32 x bf16>) -> !amx.tile<"tmm0", 16 x 16 x f32>

  // CHECK: amx.tilestored %{{.*}}, %{{.*}}, %{{.*}} : !x86.gpr<rdx>, !x86.gpr<rcx>, !amx.tile<"tmm0", 16 x 16 x f32>
  amx.tilestored %c, %stride, %td : !x86.gpr<rdx>, !x86.gpr<rcx>, !amx.tile<"tmm0", 16 x 16 x f32>

  // CHECK: amx.tilerelease
  amx.tilerelease
  return
}
