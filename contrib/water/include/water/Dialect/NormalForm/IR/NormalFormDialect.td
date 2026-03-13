// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT
#define WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT

include "mlir/IR/DialectBase.td"

def WaterNormalFormDialect : Dialect {
  let name = "water_normalform";
  let summary = "Dialect for normal form IR verification";

  let description = [{
    The WaterNormalForm dialect provides infrastructure for defining and verifying
    normal forms of IR. It uses the `water_normalform.module` operation as a
    container that specifies the expected normal forms through an array of
    attributes implementing the WaterNormalFormAttrInterface.
  }];
}

#endif // WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT
