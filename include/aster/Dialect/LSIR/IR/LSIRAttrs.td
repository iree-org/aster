//===- LSIRAttrs.td - LSIR attributes ----------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIRATTRS_TD
#define ASTER_DIALECT_LSIR_IR_LSIRATTRS_TD

include "aster/Dialect/LSIR/IR/LSIRDialect.td"
include "aster/Dialect/LSIR/IR/LSIREnums.td"

//===----------------------------------------------------------------------===//
// LSIR attributes
//===----------------------------------------------------------------------===//

defvar AnyIntTypeAttr = TypeAttrOf<AnySignlessInteger>;

#endif // ASTER_DIALECT_LSIR_IR_LSIRATTRS_TD
