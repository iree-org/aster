// RUN: water-opt %s -lower-water_normalform-module --split-input-file --verify-diagnostics

//-----------------------------------------------------------------------------
// Test that multiple top-level water_normalform.module operations are rejected.
//-----------------------------------------------------------------------------

// expected-error @below {{expected at most one top-level water_normalform.module, found 2}}
module {
  water_normalform.module [] {
    func.func @foo() {
      return
    }
  }
  water_normalform.module [] {
    func.func @bar() {
      return
    }
  }
}
