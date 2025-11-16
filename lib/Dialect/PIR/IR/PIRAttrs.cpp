//===- PIRAttrs.cpp - PIR attributes ----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/PIR/IR/PIRAttrs.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/PIR/IR/PIRAttrs.cpp.inc"
