// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMX/IR/AMXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::amx;

#include "aster/Dialect/AMX/IR/AMXDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMX/IR/AMXTypes.cpp.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMX/IR/AMXOps.cpp.inc"

void AMXDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/AMX/IR/AMXOps.cpp.inc"
      >();
}

void AMXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/AMX/IR/AMXTypes.cpp.inc"
      >();
}
