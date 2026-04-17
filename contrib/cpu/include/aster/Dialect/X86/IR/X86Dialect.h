// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86DIALECT_H
#define ASTER_DIALECT_X86_IR_X86DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aster/Dialect/X86/IR/X86Enums.h.inc"

#include "aster/Dialect/X86/IR/Interfaces/X86AsmOpInterface.h.inc"
#include "aster/Dialect/X86/IR/Interfaces/X86IsaOpInterface.h"

#include "aster/Dialect/X86/IR/X86Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Types.h.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/X86/IR/X86Ops.h.inc"

#endif // ASTER_DIALECT_X86_IR_X86DIALECT_H
