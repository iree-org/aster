//===- AllocaOpInterface.cpp - Alloca Op Interface --------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/AllocaOpInterface.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult mlir::aster::detail::verifyAllocaOpImpl(AllocaOpInterface op) {
  if (!op.getAlloca())
    return op.emitError() << "expected the alloca to be non-null";
  return success();
}

#include "aster/Interfaces/AllocaOpInterface.cpp.inc"
