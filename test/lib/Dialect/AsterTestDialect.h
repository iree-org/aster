//===- AsterTestDialect.h - test dialect ----------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT_H
#define ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT_H

#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"

#include "AsterTestDialect.h.inc"
#include "mlir/IR/Dialect.h"

// Test normal form attribute.
#define GET_ATTRDEF_CLASSES
#include "TestNormalFormAttr.h.inc"

#endif // ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT_H
