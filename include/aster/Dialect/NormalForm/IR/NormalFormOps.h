//===- NormalFormOps.h ----------------------------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_NORMALFORM_IR_NORMALFORMOPS_H
#define ASTER_DIALECT_NORMALFORM_IR_NORMALFORMOPS_H

#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/NormalForm/IR/NormalFormOps.h.inc"

#endif // ASTER_DIALECT_NORMALFORM_IR_NORMALFORMOPS_H
