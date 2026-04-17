// RUN: aster-cpu-opt %s -verify-diagnostics -split-input-file

// AVX512 op in an AVX-only module -> error.
x86.module @avx_only target_isa = #x86.isa<avx> {
  func.func @bad(
      %acc: !x86.avx512.zmm<0>,
      %lhs: !x86.avx512.zmm<1>,
      %rhs: !x86.avx512.zmm<2>) -> !x86.avx512.zmm<0> {
    // expected-error @+1 {{requires avx512 but enclosing x86.module targets avx}}
    %r = x86.avx512.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx512.zmm<0>, !x86.avx512.zmm<1>, !x86.avx512.zmm<2>)
            -> !x86.avx512.zmm<0>
    return %r : !x86.avx512.zmm<0>
  }
}

// -----

// AVX2 op in an AVX-only module -> error.
x86.module @avx_only2 target_isa = #x86.isa<avx> {
  func.func @bad(
      %acc: !x86.avx2.ymm<0>,
      %lhs: !x86.avx2.ymm<1>,
      %rhs: !x86.avx2.ymm<2>) -> !x86.avx2.ymm<0> {
    // expected-error @+1 {{requires avx2 but enclosing x86.module targets avx}}
    %r = x86.avx2.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx2.ymm<0>, !x86.avx2.ymm<1>, !x86.avx2.ymm<2>)
            -> !x86.avx2.ymm<0>
    return %r : !x86.avx2.ymm<0>
  }
}

// -----

// AVX512 op in an AVX2-only module -> error.
x86.module @avx2_only target_isa = #x86.isa<avx2> {
  func.func @bad(
      %acc: !x86.avx512.zmm<0>,
      %lhs: !x86.avx512.zmm<1>,
      %rhs: !x86.avx512.zmm<2>) -> !x86.avx512.zmm<0> {
    // expected-error @+1 {{requires avx512 but enclosing x86.module targets avx2}}
    %r = x86.avx512.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx512.zmm<0>, !x86.avx512.zmm<1>, !x86.avx512.zmm<2>)
            -> !x86.avx512.zmm<0>
    return %r : !x86.avx512.zmm<0>
  }
}

// -----

// AVX op in AVX2 module -> OK (lower ISA accepted by higher target).
// expected-no-diagnostics
x86.module @avx2_accepts_avx target_isa = #x86.isa<avx2> {
  func.func @ok(
      %acc: !x86.avx.xmm<0>,
      %lhs: !x86.avx.xmm<1>,
      %rhs: !x86.avx.xmm<2>) -> !x86.avx.xmm<0> {
    %r = x86.avx.vfmadd231ps %acc, %lhs, %rhs
        : (!x86.avx.xmm<0>, !x86.avx.xmm<1>, !x86.avx.xmm<2>)
            -> !x86.avx.xmm<0>
    return %r : !x86.avx.xmm<0>
  }
}
