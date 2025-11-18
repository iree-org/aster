//===- Passes.td - LSIR passes -----------------------------*- tablegen -*-===//
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
// LSIR passes
//===----------------------------------------------------------------------===//

def ConvertToLSIR : InterfacePass<"aster-to-lsir", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert high-level dialects to the LSIR dialect";
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect", // TODO: Use dialect interfaces.
    "mlir::aster::lsir::LSIRDialect"
  ];
}

#endif // ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_TD
