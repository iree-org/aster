//===- Passes.td - PIR passes ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_TD
#define ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// PIR passes
//===----------------------------------------------------------------------===//

def ConvertToPIR : Pass<"aster-to-pir"> {
  let summary = "Convert high-level dialects to the PIR dialect";
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect", // TODO: Use dialect interfaces.
    "mlir::aster::pir::PIRDialect"
  ];
}

#endif // ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_TD
