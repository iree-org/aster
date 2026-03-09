// RUN: aster-opt %s --amdgcn-set-normal-forms='module-forms=no_lsir_ops' | FileCheck %s --check-prefix=MODULE
// RUN: aster-opt %s --amdgcn-set-normal-forms='kernel-forms=no_scf_ops' | FileCheck %s --check-prefix=KERNEL
// RUN: aster-opt %s --amdgcn-set-normal-forms='module-forms=no_lsir_ops kernel-forms=no_scf_ops,no_cf_branches' | FileCheck %s --check-prefix=BOTH

// MODULE: amdgcn.module @test
// MODULE-SAME: attributes {normal_forms = [#amdgcn.no_lsir_ops]}
// MODULE:       kernel @k
// MODULE-NOT:   normal_forms

// KERNEL: amdgcn.module @test
// KERNEL-NOT:   normal_forms
// KERNEL:       kernel @k
// KERNEL-SAME:  normal_forms = [#amdgcn.no_scf_ops]

// BOTH: amdgcn.module @test
// BOTH-SAME: attributes {normal_forms = [#amdgcn.no_lsir_ops]}
// BOTH:       kernel @k
// BOTH-SAME:  normal_forms = [#amdgcn.no_scf_ops, #amdgcn.no_cf_branches]

amdgcn.module @test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @k {
  ^bb0:
    amdgcn.end_kernel
  }
}
