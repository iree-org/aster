//===- InstOpInterface.cpp - InstOp interface -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult mlir::aster::detail::verifyInstImpl(InstOpInterface op) {
  if (!op.isDPSInstruction())
    return success();
  ValueRange outOperands = op.getInstOuts();
  ValueRange results = op.getInstResults();
  if (outOperands.size() != results.size()) {
    return op.emitOpError()
           << "number of output operands (" << outOperands.size()
           << ") does not match number of results (" << results.size() << ")";
  }
  if (TypeRange(op.getInstOuts()) != TypeRange(op.getInstResults())) {
    return op.emitOpError()
           << "types of output operands do not match types of results";
  }
  return success();
}

bool mlir::aster::detail::isRegAllocatedImpl(InstOpInterface op) {
  /// Lambda to check if a type is allocated (not relocatable).
  auto isAllocated = +[](Type type) {
    auto regType = dyn_cast<RegisterTypeInterface>(type);
    return !regType || !regType.isRelocatable();
  };
  return llvm::all_of(TypeRange(op.getInstOuts()), isAllocated) &&
         llvm::all_of(TypeRange(op.getInstIns()), isAllocated);
}

#include "aster/Interfaces/InstOpInterface.cpp.inc"
