// RUN: aster-opt --aster-scf-pipeline="enable-rotate-stage=true rotate-stage=1" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @rotate_stage_reorderable
// CHECK-NOT:   sched.stage
// CHECK:       amdgcn.test_inst outs %{{.*}} {test_tag = 0 : i32}
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %{{.*}} {test_tag = 1 : i32}
// CHECK:       scf.for
// CHECK:       amdgcn.test_inst outs %{{.*}} ins %{{.*}} {test_tag = 2 : i32}
func.func @rotate_stage_reorderable() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32, test_tag = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = amdgcn.test_inst outs %s1 ins %a {sched.stage = 2 : i32, test_tag = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %c = amdgcn.test_inst outs %s2 ins %a {sched.stage = 1 : i32, test_tag = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
