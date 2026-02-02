//===- Passes.td - CodeGen passes --------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_CODEGEN_PASSES_TD
#define ASTER_CODEGEN_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CodeGen
//===----------------------------------------------------------------------===//

def CodeGen : InterfacePass<"aster-codegen", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert high-level dialects to the low-level dialects";
}

#endif // ASTER_CODEGEN_PASSES_TD
