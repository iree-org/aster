//===- OperandBundleOpInterface.h - OperandBundle Op Interface --*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operand bundle operation interface for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_H
#define ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace aster {
class OperandBundleOpInterface;
namespace detail {
/// Verify the operand bundle operation: single result and one or more operands.
LogicalResult verifyOperandBundleImpl(OperandBundleOpInterface op);
} // namespace detail
} // namespace aster
} // namespace mlir

#include "aster/Interfaces/OperandBundleOpInterface.h.inc"

#endif // ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_H
