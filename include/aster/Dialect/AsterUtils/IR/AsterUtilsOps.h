//===- AsterUtilsOps.h - AsterUtils dialect ops -----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations for the AsterUtils dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_H
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_H

#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h.inc"

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSOPS_H
