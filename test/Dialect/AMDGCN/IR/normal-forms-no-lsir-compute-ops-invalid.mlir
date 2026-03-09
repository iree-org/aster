// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Verify that lsir.addi (compute op) is rejected under no_lsir_compute_ops.

amdgcn.module @rejected_addi target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3>
    attributes {normal_forms = [#amdgcn.no_lsir_compute_ops]} {
  func.func @f(%dst: !amdgcn.vgpr, %a: !amdgcn.vgpr, %b: !amdgcn.vgpr) -> !amdgcn.vgpr {
    // expected-error @+1 {{normal form violation: LSIR compute/memory operations are disallowed but found: lsir.addi}}
    %result = lsir.addi i32 %dst, %a, %b : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    return %result : !amdgcn.vgpr
  }
}
