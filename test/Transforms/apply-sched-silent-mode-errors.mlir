// Tests that aster-apply-sched emits errors for missing schedules when
// silent-mode=false.

// RUN: aster-opt %s -split-input-file \
// RUN:   --pass-pipeline='builtin.module(amdgcn.module(aster-apply-sched{scheds=missing silent-mode=false}))' \
// RUN:   --verify-diagnostics

module {
  // expected-error @below {{schedule 'missing' not found}}
  amdgcn.module @default_sched_name target = <gfx942> {
    func.func @error_missing_sched() {
      return
    }
  }
}
