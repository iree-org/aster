//===- PIROps.h - PIR dialect ops -------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations for the PIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIROPS_H
#define ASTER_DIALECT_PIR_IR_PIROPS_H

#include "aster/Dialect/PIR/IR/PIRAttrs.h"
#include "aster/Dialect/PIR/IR/PIRTypes.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aster/Dialect/PIR/IR/PIROps.h.inc"

#endif // ASTER_DIALECT_PIR_IR_PIROPS_H
