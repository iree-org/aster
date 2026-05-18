//===- AMDGCNVerifiers.cpp - AMDGCN verifier implementations ----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGCN verifier functions.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// verifyISAsSupportImpl
//===----------------------------------------------------------------------===//

LogicalResult
amdgcn::verifyISAsSupportImpl(Region &region, ArrayRef<ISAVersion> isas,
                              function_ref<InFlightDiagnostic()> emitError) {
  // If no ISAs specified, verify no isa-specific AMDGCN instructions.
  if (isas.empty()) {
    LogicalResult result = success();
    region.walk([&](Operation *op) {
      if (isa<AMDGCNInstOpInterface>(op)) {
        result =
            op->emitError("target-specific AMDGCN instruction not allowed in "
                          "target-agnostic context (no ISA specified)");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  // Validate that all AMDGCNInstOpInterface operations are valid for ALL
  // the listed ISA targets.
  LogicalResult result = success();
  region.walk([&](AMDGCNInstOpInterface instOp) {
    ArrayRef<ISAVersion> opIsas = instOp.getISAVersions();
    // An instruction with no ISA list is available on all targets.
    if (opIsas.empty())
      return WalkResult::advance();
    if (!llvm::all_of(isas, [&](ISAVersion isa) {
          return llvm::is_contained(opIsas, isa);
        })) {
      result = instOp->emitError("instruction '")
               << instOp->getName()
               << "' is not valid for all specified ISA targets";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result;
}
