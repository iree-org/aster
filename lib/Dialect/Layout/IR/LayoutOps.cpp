//===- LayoutOps.cpp - Layout operations implementation -----------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/IR/LayoutOps.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::layout;

//===----------------------------------------------------------------------===//
// Dialect definition (constructor, TypeID)
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/IR/LayoutDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void LayoutDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/Layout/IR/LayoutOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/Layout/IR/LayoutOps.cpp.inc"
