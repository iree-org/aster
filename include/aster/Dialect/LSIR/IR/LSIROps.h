//===- LSIROps.h - LSIR dialect ops -----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations for the LSIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIROPS_H
#define ASTER_DIALECT_LSIR_IR_LSIROPS_H

#include "aster/Dialect/LSIR/IR/LSIRAttrs.h"
#include "aster/Dialect/LSIR/IR/LSIRTypes.h"
#include "aster/IR/InstImpl.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

#define GET_OP_CLASSES
#include "aster/Dialect/LSIR/IR/LSIROps.h.inc"

#endif // ASTER_DIALECT_LSIR_IR_LSIROPS_H
