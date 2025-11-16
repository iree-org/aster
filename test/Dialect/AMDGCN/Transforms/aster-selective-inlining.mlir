// RUN: aster-opt %s --aster-selective-inlining | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: aster-opt %s --aster-selective-inlining="allow-scheduled-calls=true" | FileCheck %s --check-prefix=CHECK-ALLOW

func.func @helper_normal(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %1 : !amdgcn.vgpr<1>
}

func.func @helper_scheduled(%x: !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1> {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %0, %x : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %1 : !amdgcn.vgpr<1>
}

// Test selective inlining: default behavior (should NOT inline scheduled calls)
// CHECK-DEFAULT-LABEL: func.func @main
//   CHECK-DEFAULT-NOT:   call @helper_normal
//       CHECK-DEFAULT:   call @helper_scheduled

// Test selective inlining: with --allow-scheduled-calls (should inline all)
// CHECK-ALLOW-LABEL: func.func @main
//   CHECK-ALLOW-NOT:     call @helper_normal
//   CHECK-ALLOW-NOT:     call @helper_scheduled
func.func @main(%arg: !amdgcn.vgpr<0>) -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
  %result_normal = call @helper_normal(%arg) : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  %result_scheduled = call @helper_scheduled(%arg) {sched.delay = 0 : i64, sched.rate = 1 : i64} : (!amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>
  return %result_normal, %result_scheduled : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
}
