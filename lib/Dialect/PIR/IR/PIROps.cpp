//===- PIROps.cpp - PIR operations ------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Dialect/PIR/IR/PIRAttrs.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/IR/PIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

//===----------------------------------------------------------------------===//
// PIR dialect
//===----------------------------------------------------------------------===//

void PIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/PIR/IR/PIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/PIR/IR/PIRAttrs.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// TypeCastOp
//===----------------------------------------------------------------------===//

OpFoldResult TypeCastOp::fold(FoldAdaptor adaptor) {
  if (getValue().getType() == getDst().getType())
    return getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// PIR IncGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/PIR/IR/PIROps.cpp.inc"

#include "aster/Dialect/PIR/IR/PIRDialect.cpp.inc"
