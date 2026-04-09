// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Verify that lsir.cmpi is rejected under no_lsir_control_ops.

amdgcn.module @rejected_cmpi target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3>
    attributes {normal_forms = [#amdgcn.no_lsir_control_ops]} {
  func.func @f(%dst: !amdgcn.scc, %a: !amdgcn.sgpr, %b: !amdgcn.sgpr) -> !amdgcn.scc {
    // expected-error @+1 {{normal form violation: LSIR control-flow operations are disallowed but found: lsir.cmpi}}
    %cmp = lsir.cmpi i32 slt %dst, %a, %b : !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
    return %cmp : !amdgcn.scc
  }
}
