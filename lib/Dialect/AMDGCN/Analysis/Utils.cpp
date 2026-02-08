//===- Utils.cpp - AMDGCN Analysis Utilities ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/Utils.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/RegisterType.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;

FailureOr<ValueRange> mlir::aster::amdgcn::getAllocasOrFailure(Value value) {
  // If the value is not a register, return an empty range.
  if (!isa<RegisterTypeInterface>(value.getType()))
    return ValueRange{};

  // Handle alloca operations.
  if (auto allocaOp = value.getDefiningOp<AllocaOp>())
    return allocaOp->getResults();

  // Handle make register range operations.
  if (auto makeRegisterRangeOp = value.getDefiningOp<MakeRegisterRangeOp>()) {
    for (Value input : makeRegisterRangeOp.getInputs()) {
      if (!isa_and_nonnull<AllocaOp>(input.getDefiningOp()))
        return failure();
    }
    return makeRegisterRangeOp.getInputs();
  }

  // Fail in all other cases.
  return failure();
}
