//===- LSIRAttrs.h - LSIR dialect attributes --------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes for the LSIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIRATTRS_H
#define ASTER_DIALECT_LSIR_IR_LSIRATTRS_H

#include "aster/Dialect/LSIR/IR/LSIREnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/LSIR/IR/LSIRAttrs.h.inc"

#endif // ASTER_DIALECT_LSIR_IR_LSIRATTRS_H
