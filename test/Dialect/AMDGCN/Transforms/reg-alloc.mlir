// RUN: aster-opt %s --amdgcn-reg-alloc | FileCheck %s
// RUN: aster-opt %s --amdgcn-reg-alloc="mode=full optimize=false" | FileCheck %s --check-prefix=CHECK-FULL

amdgcn.module @reg_alloc target = <gfx942> isa = <cdna3> {
  // CHECK-LABEL: reg_alloc
  // CHECK-FULL-LABEL: reg_alloc
  func.func private @rand() -> i1
  // CHECK-NOT: alloca : !amdgcn.vgpr<1>
  // CHECK: alloca : !amdgcn.vgpr<0>
  // CHECK-FULL-DAG: alloca : !amdgcn.vgpr<1>
  // CHECK-FULL-DAG: alloca : !amdgcn.vgpr<0>
  kernel @reg_alloc {
    %0 = func.call @rand() : () -> i1
    %1 = alloca : !amdgcn.vgpr
    %2 = alloca : !amdgcn.vgpr
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%1 : !amdgcn.vgpr)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%2 : !amdgcn.vgpr)
  ^bb3(%3: !amdgcn.vgpr):  // 2 preds: ^bb1, ^bb2
    %4 = test_inst outs %3 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    end_kernel
  }
}
