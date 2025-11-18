//===- RegisterConstraints.cpp - Register constraints -----------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/RegisterConstraints.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::aster;

FailureOr<RegisterConstraints> RegisterConstraints::create(Operation *op) {
  if (!op)
    return failure();
  RegisterConstraints analysis;
  WalkResult wR = op->walk([&](lsir::RegConstraintOp cOp) {
    Value target = cOp.getInput();
    Attribute constraint = cOp.getKind();
    // If there is already a constraint for the value, ensure they are
    // consistent.
    if (Attribute existing = analysis.constraints.lookup(target)) {
      if (existing != constraint) {
        cOp.emitError() << "Conflicting register constraints"
                        << ": " << existing << " vs " << constraint;
        return WalkResult::interrupt();
      }
    } else {
      analysis.constraints.insert({target, constraint});
    }
    return WalkResult::advance();
  });
  if (wR.wasInterrupted())
    return failure();
  return analysis;
}
