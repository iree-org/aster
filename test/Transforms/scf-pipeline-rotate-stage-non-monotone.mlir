// RUN: aster-opt --aster-scf-pipeline="enable-rotate-stage=true rotate-stage=1" \
// RUN:   --verify-diagnostics --split-input-file %s

func.func @rotate_stage_non_monotone() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %s3 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // expected-error @+1 {{cannot be made stage-monotone}}
  scf.for %i = %c0 to %c4 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // expected-note @+1 {{value defined here}}
    %b = amdgcn.test_inst outs %s1 ins %a {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // expected-note @+1 {{no monotone order exists}}
    %c = amdgcn.test_inst outs %s2 ins %b {sched.stage = 0 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %d = amdgcn.test_inst outs %s3 ins %c {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}

// -----

func.func @rotate_stage_non_monotone_three_stage() {
  %s0 = amdgcn.alloca : !amdgcn.vgpr
  %s1 = amdgcn.alloca : !amdgcn.vgpr
  %s2 = amdgcn.alloca : !amdgcn.vgpr
  %s3 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @+1 {{cannot be made stage-monotone}}
  scf.for %i = %c0 to %c8 step %c1 {
    %a = amdgcn.test_inst outs %s0 {sched.stage = 0 : i32} : (!amdgcn.vgpr) -> !amdgcn.vgpr
    // expected-note @+1 {{value defined here}}
    %b = amdgcn.test_inst outs %s1 ins %a {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    // expected-note @+1 {{no monotone order exists}}
    %c = amdgcn.test_inst outs %s2 ins %b {sched.stage = 1 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
    %d = amdgcn.test_inst outs %s3 ins %c {sched.stage = 2 : i32} : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  }
  return
}
