// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

def TestWaveDialectFunctionsPass : Pass<"water-test-wave-dialect-functions"> {
  let summary =
    "Test functionalities of Wave dialect exercised elsewhere indirectly";
  let description = [{
    Tests various functionalities of the Wave dialect directly before they
    can be called from elsewhere.

    * Construct attributes and types from primitive elements using `getChecked` to
      trigger error emission instead of assertions.

    * Convert the IterateOp between isolated and non-isolated forms.
  }];

  let dependentDialects = ["::wave::WaveDialect"];
}
