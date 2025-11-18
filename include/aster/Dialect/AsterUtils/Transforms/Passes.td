//===- Passes.td - AsterUtils passes -----------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD
#define ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AsterUtils passes
//===----------------------------------------------------------------------===//

def InlineExecuteRegion : Pass<"aster-inline-execute-region"> {
  let summary = "Inline aster_utils.execute_region operations";
  let description = [{
    This pass inlines `aster_utils.execute_region` operations by replacing
    them with the contents of their regions. The yield operation results
    are replaced with the yielded values.
  }];
}

def WrapCallsWithExecuteRegion : Pass<"aster-wrap-calls-with-execute-region"> {
  let summary = "Wrap function calls with aster_utils.execute_region";
  let description = [{
    This pass wraps function calls with `aster_utils.execute_region` operations.
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
}

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD
