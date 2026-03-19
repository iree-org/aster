//===- LayoutAttrs.h - Layout attributes --------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_H
#define ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/Layout/IR/LayoutAttrs.h.inc"

#endif // ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_H
