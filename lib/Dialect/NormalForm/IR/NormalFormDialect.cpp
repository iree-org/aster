//===- NormalFormDialect.cpp - dialect definition -------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "aster/Dialect/NormalForm/IR/NormalFormOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

#include "aster/Dialect/NormalForm/IR/NormalFormDialect.cpp.inc"

void normalform::NormalFormDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/NormalForm/IR/NormalFormOps.cpp.inc"
      >();
}
