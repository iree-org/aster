// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_TEST_LIB_DIALECT_TESTNORMALFORMATTR
#define WATER_TEST_LIB_DIALECT_TESTNORMALFORMATTR

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "water/Dialect/NormalForm/IR/NormalFormInterfaces.td"

// Reuse the WaterTestDialect definition.
def WaterTestDialect : Dialect {
  let name = "water_test";
  let cppNamespace = "::mlir::water::test";
  let useDefaultAttributePrinterParser = 1;
}

//-----------------------------------------------------------------------------
// Test normal form attributes for testing WaterNormalFormAttrInterface.
// Each attribute tests a single invariant.
//-----------------------------------------------------------------------------

def NoIndexTypesAttr : AttrDef<WaterTestDialect, "NoIndexTypes",
    [DeclareAttrInterfaceMethods<WaterNormalFormAttrInterface, ["verifyType"]>]> {
  let mnemonic = "no_index_types";
  let summary = "Normal form attribute that prohibits index types";
  let description = [{
    A test attribute implementing WaterNormalFormAttrInterface that verifies
    no index types are used in operations, block arguments, or results.
  }];
}

def NoInvalidOpsAttr : AttrDef<WaterTestDialect, "NoInvalidOps",
    [DeclareAttrInterfaceMethods<WaterNormalFormAttrInterface, ["verifyOperation"]>]> {
  let mnemonic = "no_invalid_ops";
  let summary = "Normal form attribute that prohibits division operations";
  let description = [{
    A test attribute implementing WaterNormalFormAttrInterface that verifies
    no division operations (arith.divf, arith.divsi, arith.divui) are present.
  }];
}

def NoInvalidAttrsAttr : AttrDef<WaterTestDialect, "NoInvalidAttrs",
    [DeclareAttrInterfaceMethods<WaterNormalFormAttrInterface, ["verifyAttribute"]>]> {
  let mnemonic = "no_invalid_attrs";
  let summary = "Normal form attribute that prohibits 'invalid' string values";
  let description = [{
    A test attribute implementing WaterNormalFormAttrInterface that verifies
    no string attributes have the value "invalid".
  }];
}

def NoForbiddenSymbolsAttr : AttrDef<WaterTestDialect, "NoForbiddenSymbols",
    [DeclareAttrInterfaceMethods<WaterNormalFormAttrInterface, ["verifyAttribute"]>]> {
  let mnemonic = "no_forbidden_symbols";
  let summary = "Normal form attribute that prohibits WaveSymbolAttr with name 'forbidden'";
  let description = [{
    A test attribute implementing WaterNormalFormAttrInterface that verifies
    no WaveSymbolAttr has the name "forbidden". This is used to test that
    attributes embedded within types (like WaveTensorType's shape) are verified.
  }];
}

#endif // WATER_TEST_LIB_DIALECT_TESTNORMALFORMATTR
