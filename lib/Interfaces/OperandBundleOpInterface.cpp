//===- OperandBundleOpInterface.cpp - OperandBundle Op Interface ----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operand bundle operation interface.
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/OperandBundleOpInterface.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult
mlir::aster::detail::verifyOperandBundleImpl(OperandBundleOpInterface op) {
  Operation *operation = op.getOperation();
  if (operation->getNumResults() != 1) {
    return operation->emitError()
           << "OperandBundleOpInterface requires exactly one result, got "
           << operation->getNumResults();
  }
  if (operation->getNumOperands() < 1) {
    return operation->emitError()
           << "OperandBundleOpInterface requires one or more operands, got "
           << operation->getNumOperands();
  }
  return success();
}

#include "aster/Interfaces/OperandBundleOpInterface.cpp.inc"
