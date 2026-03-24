// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Verify that lsir.cmpi (control op) is allowed under no_lsir_compute_ops.

amdgcn.module @allowed_cmpi target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3>
    attributes {normal_forms = [#amdgcn.no_lsir_compute_ops]} {
  func.func @f(%dst: !amdgcn.scc, %a: !amdgcn.sgpr, %b: !amdgcn.sgpr) -> !amdgcn.scc {
    %cmp = lsir.cmpi i32 slt %dst, %a, %b : !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
    return %cmp : !amdgcn.scc
  }
}

// -----

// Verify that lsir.copy (regalloc primitive) is allowed under no_lsir_compute_ops.

amdgcn.module @allowed_copy target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3>
    attributes {normal_forms = [#amdgcn.no_lsir_compute_ops]} {
  func.func @f(%dst: !amdgcn.vgpr, %src: !amdgcn.vgpr) -> !amdgcn.vgpr {
    %result = lsir.copy %dst, %src : !amdgcn.vgpr, !amdgcn.vgpr
    return %result : !amdgcn.vgpr
  }
}
