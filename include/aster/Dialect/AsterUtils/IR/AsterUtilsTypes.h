//===- AsterUtilsTypes.h - AsterUtils dialect types -------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the AsterUtils dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_H
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_H

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h.inc"

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSTYPES_H
