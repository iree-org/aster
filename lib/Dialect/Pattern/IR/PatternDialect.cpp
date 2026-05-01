//===- PatternDialect.cpp - dialect definition ----------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Pattern/IR/PatternDialect.h"
#include "aster/Dialect/Pattern/IR/PatternOps.h"

#include "mlir/IR/Dialect.h"

using namespace mlir;

#include "aster/Dialect/Pattern/IR/PatternDialect.cpp.inc"

void aster::pattern::PatternDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/Pattern/IR/PatternOps.cpp.inc"
      >();
}
