//===- AsterTestDialect.td - test dialect ---------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

#ifndef ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT
#define ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT

def AsterTestDialect : Dialect {
  let name = "aster_test";
  let summary = "Dialect for testing Aster";
  let cppNamespace = "::mlir::aster::test";
  let useDefaultAttributePrinterParser = 1;
}

#endif // ASTER_TEST_LIB_DIALECT_WATERTESTDIALECT
