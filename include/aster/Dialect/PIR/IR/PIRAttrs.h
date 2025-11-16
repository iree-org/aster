//===- PIRAttrs.h - PIR dialect attributes ----------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes for the PIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIRATTRS_H
#define ASTER_DIALECT_PIR_IR_PIRATTRS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/PIR/IR/PIRAttrs.h.inc"

#endif // ASTER_DIALECT_PIR_IR_PIRATTRS_H
