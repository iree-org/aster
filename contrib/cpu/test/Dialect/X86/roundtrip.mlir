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
