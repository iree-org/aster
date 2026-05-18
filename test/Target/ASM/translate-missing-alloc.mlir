// RUN: not aster-translate --split-input-file %s --mlir-to-asm 2>&1 | FileCheck %s

// Unallocated VGPR type is rejected before translation.
// CHECK: normal form violation: all registers must have allocated semantics but found: '!amdgcn.vgpr<?>'

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr<?>
    amdgcn.end_kernel
  }
}

// -----

// Unallocated SGPR type is rejected before translation.
// CHECK: normal form violation: all registers must have allocated semantics but found: '!amdgcn.sgpr<?>'

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.sgpr<?>
    amdgcn.end_kernel
  }
}

// -----

// Value-semantic (unresolved) VGPR type is rejected before translation.
// CHECK: normal form violation: all registers must have allocated semantics but found: '!amdgcn.vgpr'

amdgcn.module @mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^bb0:
    %0 = amdgcn.alloca : !amdgcn.vgpr
    amdgcn.end_kernel
  }
}
