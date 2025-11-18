//===- LSIRTypes.h - LSIR dialect types -------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the LSIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIRTYPES_H
#define ASTER_DIALECT_LSIR_IR_LSIRTYPES_H

#include "aster/Interfaces/DependentOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/LSIR/IR/LSIRTypes.h.inc"

#endif // ASTER_DIALECT_LSIR_IR_LSIRTYPES_H
