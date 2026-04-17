// RUN: aster-cpu-opt %s | aster-cpu-opt | FileCheck %s

// CHECK-LABEL: x86.module @avx_module target_isa = <avx>
x86.module @avx_module target_isa = #x86.isa<avx> {
  func.func @fma_avx(
      %acc: !x86.avx.xmm<0>,
      %lhs: !x86.avx.xmm<1>,
      %rhs: !x86.avx.xmm<2>) -> !x86.avx.xmm<0> {
    // CHECK: x86.avx.vfmadd231ps
    %r = x86.avx.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>)
            -> !x86.avx.xmm<0>
    return %r : !x86.avx.xmm<0>
  }
}

// CHECK-LABEL: x86.module @avx2_module target_isa = <avx2>
x86.module @avx2_module target_isa = #x86.isa<avx2> {
  func.func @fma_avx(%acc: !x86.avx.xmm<0>, %lhs: !x86.avx.xmm<1>,
                      %rhs: !x86.avx.xmm<2>) -> !x86.avx.xmm<0> {
    // CHECK: x86.avx.vfmadd231ps
    %r = x86.avx.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>)
            -> !x86.avx.xmm<0>
    return %r : !x86.avx.xmm<0>
  }
  func.func @fma_avx2(%acc: !x86.avx2.ymm<0>, %lhs: !x86.avx2.ymm<1>,
                       %rhs: !x86.avx2.ymm<2>) -> !x86.avx2.ymm<0> {
    // CHECK: x86.avx2.vfmadd231ps
    %r = x86.avx2.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx2.ymm<0>, !x86.avx2.ymm<1>, !x86.avx2.ymm<2>)
            -> !x86.avx2.ymm<0>
    return %r : !x86.avx2.ymm<0>
  }
}

// CHECK-LABEL: x86.module @avx512_module target_isa = <avx512>
x86.module @avx512_module target_isa = #x86.isa<avx512> {
  func.func @fma_avx(%acc: !x86.avx.xmm<0>, %lhs: !x86.avx.xmm<1>,
                      %rhs: !x86.avx.xmm<2>) -> !x86.avx.xmm<0> {
    %r = x86.avx.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>)
            -> !x86.avx.xmm<0>
    return %r : !x86.avx.xmm<0>
  }
  func.func @fma_avx2(%acc: !x86.avx2.ymm<0>, %lhs: !x86.avx2.ymm<1>,
                       %rhs: !x86.avx2.ymm<2>) -> !x86.avx2.ymm<0> {
    %r = x86.avx2.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx2.ymm<0>, !x86.avx2.ymm<1>, !x86.avx2.ymm<2>)
            -> !x86.avx2.ymm<0>
    return %r : !x86.avx2.ymm<0>
  }
  func.func @fma_avx512(%acc: !x86.avx512.zmm<0>, %lhs: !x86.avx512.zmm<1>,
                         %rhs: !x86.avx512.zmm<2>) -> !x86.avx512.zmm<0> {
    // CHECK: x86.avx512.vfmadd231ps
    %r = x86.avx512.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx512.zmm<0>, !x86.avx512.zmm<1>, !x86.avx512.zmm<2>)
            -> !x86.avx512.zmm<0>
    return %r : !x86.avx512.zmm<0>
  }
}
