// RUN: aster-opt %s --pass-pipeline='builtin.module(canonicalize,amdgcn.module(amdgcn-convert-waits))' | aster-translate --mlir-to-asm | FileCheck %s --check-prefix=ASM
// RUN: aster-opt %s --pass-pipeline='builtin.module(canonicalize,amdgcn.module(amdgcn-convert-waits))' \
// RUN:   | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// token_barrier lowers to the gfx1250 split barrier in emitted asm.

//      ASM: s_barrier_signal -1
// ASM-NEXT: s_barrier_wait -1
amdgcn.module @cross_wave_barrier_gfx1250_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cross_wave_barrier_gfx1250 attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %fence = amdgcn.token_barrier scope(<workgroup>)
    amdgcn.end_kernel
  }
}
