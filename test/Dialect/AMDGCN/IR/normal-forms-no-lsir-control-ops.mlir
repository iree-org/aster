// RUN: aster-opt %s --split-input-file --verify-diagnostics

// Verify that lsir.addi (compute op) is allowed under no_lsir_control_ops.

amdgcn.module @allowed_addi target = #amdgcn.target<gfx942>
    attributes {normal_forms = [#amdgcn.no_lsir_control_ops]} {
  func.func @f(%dst: !amdgcn.vgpr, %a: !amdgcn.vgpr, %b: !amdgcn.vgpr) -> !amdgcn.vgpr {
    %result = lsir.addi i32 %dst, %a, %b : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
    return %result : !amdgcn.vgpr
  }
}
