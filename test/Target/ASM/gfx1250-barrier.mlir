// RUN: aster-opt %s | aster-translate --mlir-to-asm | FileCheck %s --check-prefix=ASM
// RUN: aster-opt %s | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// GFX12.5 uses s_barrier_signal/wait instead of s_barrier.

//      ASM: s_barrier_signal -1
// ASM-NEXT: s_barrier_wait -1

amdgcn.module @sbarrier_gfx1250_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @s_barrier_gfx1250 attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    amdgcn.s_barrier_signal -1
    amdgcn.s_barrier_wait -1
    amdgcn.end_kernel
  }
}
