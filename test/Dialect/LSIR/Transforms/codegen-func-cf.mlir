// RUN: aster-opt %s --aster-codegen | FileCheck %s

// CHECK-LABEL: amdgcn.module @test
// CHECK:   func.func @loop_func
// CHECK:     cf.cond_br %{{.*}}, ^bb1(%{{.*}} : !amdgcn.sgpr), ^bb2
// CHECK:   ^bb1(%{{.*}}: !amdgcn.sgpr):
// CHECK:     cf.cond_br %{{.*}}, ^bb1(%{{.*}} : !amdgcn.sgpr), ^bb2
// CHECK:   ^bb2:

amdgcn.module @test target = <gfx942> {
  func.func @loop_func(%arg0: i32, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cmp_init = arith.cmpi slt, %c0, %n : i32
    cf.cond_br %cmp_init, ^bb1(%c0 : i32), ^bb2
  ^bb1(%iv: i32):
    %next = arith.addi %iv, %c1 : i32
    %cmp = arith.cmpi slt, %next, %n : i32
    cf.cond_br %cmp, ^bb1(%next : i32), ^bb2
  ^bb2:
    return
  }
}
