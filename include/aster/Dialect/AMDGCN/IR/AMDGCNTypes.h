//===- AMDGCNTypes.h - AMDGCN Types -----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Types.h"

namespace mlir::aster::amdgcn {
/// Get the register kind as an integer from the given register type.
/// This call asserts if type is not an AMD register.
RegisterKind getRegisterKind(RegisterTypeInterface type);
} // namespace mlir::aster::amdgcn

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h.inc"

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H
