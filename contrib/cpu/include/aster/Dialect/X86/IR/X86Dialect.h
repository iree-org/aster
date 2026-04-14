// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86DIALECT_H
#define ASTER_DIALECT_X86_IR_X86DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "aster/Dialect/X86/IR/X86Enums.h.inc"

#include "aster/Dialect/X86/IR/X86Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Types.h.inc"

#endif // ASTER_DIALECT_X86_IR_X86DIALECT_H
