// RUN: aster-opt --aster-scf-pipeline="enable-rotate-stage=true rotate-stage=5" \
// RUN:   --verify-diagnostics --split-input-file %s

func.func @rotate_stage_out_of_range() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // expected-error @+1 {{rotate-stage must be in [0, maxStage], got 5 for maxStage 1}}
  scf.for %i = %c0 to %c3 step %c1 {
    %p = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %q = amdgcn.test_inst outs %s1 ins %p {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----

func.func @rotate_stage_not_in_body() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{rotate-stage 5 does not appear in the loop body}}
  scf.for %i = %c0 to %c8 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %b = amdgcn.test_inst outs %s1 ins %a {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %c = amdgcn.test_inst outs %s2 ins %b {sched.stage = 6 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
