//===- LayoutOps.h - Layout operations ----------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_H
#define ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_H

#include "aster/Dialect/Layout/IR/LayoutAttrs.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/Layout/IR/LayoutOps.h.inc"

#endif // ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_H
