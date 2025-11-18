//===- AsterUtilsAttrs.td - AsterUtils attributes ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD

include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.td"
include "aster/Dialect/AsterUtils/IR/AsterUtilsEnums.td"

//===----------------------------------------------------------------------===//
// GPU dimension identifier.
//===----------------------------------------------------------------------===//

def DimAttr : AsterUtils_EnumAttr<"dim", DimEnum, []> {
  let description = [{
    AsterUtils gpu dimension identifier attribute.
  }];
}

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_TD
