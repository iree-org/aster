// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK-LABEL: test_setprio_inc_wg_asm:
//       CHECK: s_setprio_inc_wg 100
//  CHECK-NEXT: s_setprio_inc_wg 0
//       CHECK: s_endpgm

amdgcn.module @test_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @test_setprio_inc_wg_asm {
    s_setprio_inc_wg 100
    s_setprio_inc_wg 0
    amdgcn.end_kernel
  }
}
