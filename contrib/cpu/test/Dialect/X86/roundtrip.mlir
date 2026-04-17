// RUN: aster-cpu-opt %s | aster-cpu-opt | FileCheck %s

// CHECK-LABEL: func.func @avx_fma
func.func @avx_fma(
    %acc: !x86.avx.xmm<0>,
    %lhs: !x86.avx.xmm<1>,
    %rhs: !x86.avx.xmm<2>) -> !x86.avx.xmm<0> {
  // CHECK: x86.avx.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>) -> !x86.avx.xmm<0>
  %r = x86.avx.vfmadd231ps %acc, %lhs, %rhs
      : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>)
          -> !x86.avx.xmm<0>
  return %r : !x86.avx.xmm<0>
}

// CHECK-LABEL: func.func @avx2_fma
func.func @avx2_fma(
    %acc: !x86.avx2.ymm<0>,
    %lhs: !x86.avx2.ymm<1>,
    %rhs: !x86.avx2.ymm<2>) -> !x86.avx2.ymm<0> {
  // CHECK: x86.avx2.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx2.ymm<0>, !x86.avx2.ymm<1>, !x86.avx2.ymm<2>) -> !x86.avx2.ymm<0>
  %r = x86.avx2.vfmadd231ps %acc, %lhs, %rhs
      : (!x86.avx2.ymm<0>, !x86.avx2.ymm<1>, !x86.avx2.ymm<2>)
          -> !x86.avx2.ymm<0>
  return %r : !x86.avx2.ymm<0>
}

// CHECK-LABEL: func.func @avx512_fma
func.func @avx512_fma(
    %acc: !x86.avx512.zmm<0>,
    %lhs: !x86.avx512.zmm<1>,
    %rhs: !x86.avx512.zmm<2>) -> !x86.avx512.zmm<0> {
  // CHECK: x86.avx512.vfmadd231ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avx512.zmm<0>, !x86.avx512.zmm<1>, !x86.avx512.zmm<2>) -> !x86.avx512.zmm<0>
  %r = x86.avx512.vfmadd231ps %acc, %lhs, %rhs
      : (!x86.avx512.zmm<0>, !x86.avx512.zmm<1>, !x86.avx512.zmm<2>)
          -> !x86.avx512.zmm<0>
  return %r : !x86.avx512.zmm<0>
}

// CHECK-LABEL: func.func @avx_load_store
func.func @avx_load_store(%ptr: !x86.gpr<rdi>) {
  // CHECK: x86.avx.load vmovaps %{{.*}} : !x86.gpr<rdi> -> !x86.avx.xmm<0>
  %v = x86.avx.load vmovaps %ptr : !x86.gpr<rdi> -> !x86.avx.xmm<0>
  // CHECK: x86.avx.store vmovaps %{{.*}}, %{{.*}} : !x86.gpr<rdi>, !x86.avx.xmm<0>
  x86.avx.store vmovaps %ptr, %v : !x86.gpr<rdi>, !x86.avx.xmm<0>
  return
}

// CHECK-LABEL: func.func @amx_roundtrip
func.func @amx_roundtrip(
    %a: !x86.gpr<rdi>,
    %b: !x86.gpr<rsi>,
    %c: !x86.gpr<rdx>,
    %stride: !x86.gpr<rcx>) {
  // CHECK: x86.amx.ldtilecfg
  x86.amx.ldtilecfg
  // CHECK: x86.amx.tileloadd %{{.*}}, %{{.*}} : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !x86.amx.tmm<1, 16 x 32 x bf16>
  %ta = x86.amx.tileloadd %a, %stride : (!x86.gpr<rdi>, !x86.gpr<rcx>) -> !x86.amx.tmm<1, 16 x 32 x bf16>
  // CHECK: x86.amx.tileloadd %{{.*}}, %{{.*}} : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !x86.amx.tmm<2, 16 x 32 x bf16>
  %tb = x86.amx.tileloadd %b, %stride : (!x86.gpr<rsi>, !x86.gpr<rcx>) -> !x86.amx.tmm<2, 16 x 32 x bf16>
  // CHECK: x86.amx.tilezero : !x86.amx.tmm<0, 16 x 16 x f32>
  %tc = x86.amx.tilezero : !x86.amx.tmm<0, 16 x 16 x f32>
  // CHECK: x86.amx.tdpbf16ps %{{.*}}, %{{.*}}, %{{.*}} : (!x86.amx.tmm<0, 16 x 16 x f32>, !x86.amx.tmm<1, 16 x 32 x bf16>, !x86.amx.tmm<2, 16 x 32 x bf16>) -> !x86.amx.tmm<0, 16 x 16 x f32>
  %td = x86.amx.tdpbf16ps %tc, %ta, %tb
      : (!x86.amx.tmm<0, 16 x 16 x f32>, !x86.amx.tmm<1, 16 x 32 x bf16>, !x86.amx.tmm<2, 16 x 32 x bf16>) -> !x86.amx.tmm<0, 16 x 16 x f32>
  // CHECK: x86.amx.tilestored %{{.*}}, %{{.*}}, %{{.*}} : !x86.gpr<rdx>, !x86.gpr<rcx>, !x86.amx.tmm<0, 16 x 16 x f32>
  x86.amx.tilestored %c, %stride, %td : !x86.gpr<rdx>, !x86.gpr<rcx>, !x86.amx.tmm<0, 16 x 16 x f32>
  // CHECK: x86.amx.tilerelease
  x86.amx.tilerelease
  return
}
