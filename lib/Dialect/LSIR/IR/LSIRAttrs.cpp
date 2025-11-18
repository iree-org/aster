//===- LSIRAttrs.cpp - LSIR attributes --------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/LSIR/IR/LSIRAttrs.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::lsir;

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/LSIR/IR/LSIRAttrs.cpp.inc"

void LSIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/LSIR/IR/LSIRAttrs.cpp.inc"
      >();
}
