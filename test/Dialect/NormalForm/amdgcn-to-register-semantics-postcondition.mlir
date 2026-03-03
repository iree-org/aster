// Demonstrates that #amdgcn.no_value_semantic_registers is a post-condition
// of the amdgcn-to-register-semantics pass.
//
// Step 1: Run the pass on IR with value-semantic registers.
// Step 2: Wrap the output in a normalform.module and re-verify.
//
// RUN: aster-opt --amdgcn-to-register-semantics %s \
// RUN:   | sed '1s/^module/normalform.module [#amdgcn.no_value_semantic_registers]/' \
// RUN:   | aster-opt --verify-diagnostics

func.func @value_to_unallocated() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = lsir.copy %1, %0 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
  amdgcn.test_inst ins %3, %2 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
  func.return
}

func.func @mixed_types(%arg: i32) -> f32 {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.sgpr
  %cst = arith.constant 0.0 : f32
  return %cst : f32
}
