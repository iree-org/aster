// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_AMXDIALECT_H
#define ASTER_DIALECT_AMX_IR_AMXDIALECT_H

// Load-bearing: mlir-tblgen auto-adds ::mlir::BytecodeOpInterface::Trait to
// every generated Op<> base; removing this include breaks Op instantiation.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.h"
// AMXOps reference operands of type !x86.gpr.
#include "aster/Dialect/X86/IR/X86Dialect.h"

#include "aster/Dialect/AMX/IR/AMXDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMX/IR/AMXTypes.h.inc"

#define GET_OP_CLASSES
#include "aster/Dialect/AMX/IR/AMXOps.h.inc"

#endif // ASTER_DIALECT_AMX_IR_AMXDIALECT_H
