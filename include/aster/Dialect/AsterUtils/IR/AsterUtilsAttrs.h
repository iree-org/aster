//===- AsterUtilsAttrs.h - AsterUtils dialect attributes --------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes for the AsterUtils dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_H
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_H

#include "aster/Dialect/AsterUtils/IR/AsterUtilsEnums.h"
#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h.inc"

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSATTRS_H
