//===- AsterUtilsTypes.cpp - AsterUtils types -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

void AsterUtilsDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.cpp.inc"
