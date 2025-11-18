//===- PIRTypes.h - PIR dialect types ---------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the PIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIRTYPES_H
#define ASTER_DIALECT_PIR_IR_PIRTYPES_H

#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/PIR/IR/PIRTypes.h.inc"

namespace mlir::aster {
namespace pir {
Type getUnderlyingType(Type type);
} // namespace pir
} // namespace mlir::aster

#endif // ASTER_DIALECT_PIR_IR_PIRTYPES_H
