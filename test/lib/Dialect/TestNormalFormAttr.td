//===- TestNormalFormAttrs.td - definitions for NormalForm tests ----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TEST_LIB_DIALECT_TESTNORMALFORMATTR
#define ASTER_TEST_LIB_DIALECT_TESTNORMALFORMATTR

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.td"
include "AsterTestDialect.td"

//-----------------------------------------------------------------------------
// Test normal form attributes for testing NormalFormAttrInterface.
// Each attribute tests a single invariant.
//-----------------------------------------------------------------------------

def NoIndexTypesAttr : AttrDef<AsterTestDialect, "NoIndexTypes",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyType"]>]> {
  let mnemonic = "no_index_types";
  let summary = "Normal form attribute that prohibits index types";
  let description = [{
    A test attribute implementing NormalFormAttrInterface that verifies
    no index types are used in operations, block arguments, or results.
  }];
}

def NoInvalidOpsAttr : AttrDef<AsterTestDialect, "NoInvalidOps",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let mnemonic = "no_invalid_ops";
  let summary = "Normal form attribute that prohibits division operations";
  let description = [{
    A test attribute implementing NormalFormAttrInterface that verifies
    no division operations (arith.divf, arith.divsi, arith.divui) are present.
  }];
}

def NoInvalidAttrsAttr : AttrDef<AsterTestDialect, "NoInvalidAttrs",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyAttribute"]>]> {
  let mnemonic = "no_invalid_attrs";
  let summary = "Normal form attribute that prohibits 'invalid' string values";
  let description = [{
    A test attribute implementing NormalFormAttrInterface that verifies
    no string attributes have the value "invalid".
  }];
}

#endif // ASTER_TEST_LIB_DIALECT_TESTNORMALFORMATTR
