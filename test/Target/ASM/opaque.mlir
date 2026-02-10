// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK-LABEL:Module: mod
// CHECK: v_add_f32 v3, v2, v3
// CHECK: v_add_f32 v3, v2, v3 offset(0) neg
amdgcn.module @mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @opaque {
    %0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %1 = amdgcn.alloca : !amdgcn.vgpr<3>
    %2 = amdgcn.alloca : !amdgcn.vgpr<3>
    opaque "v_add_f32" outs(%2) ins(%0, %1) : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<3>)
    opaque "v_add_f32" outs(%2) ins(%0, %1) modifiers("offset(0)", neg) : (!amdgcn.vgpr<3>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> (!amdgcn.vgpr<3>)
    amdgcn.end_kernel
  }
}
