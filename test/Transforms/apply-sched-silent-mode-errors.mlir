// Tests that aster-apply-sched emits errors for missing schedules when
// silent-mode=false.

// RUN: aster-opt %s -split-input-file \
// RUN:   --aster-apply-sched="scheds=missing silent-mode=false" \
// RUN:   --verify-diagnostics

// expected-error @below {{schedule 'missing' not found}}
module {
  func.func @error_missing_sched() {
    return
  }
}
