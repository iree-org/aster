//===- Utils.h - AMDGCN Analysis Utilities -----------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_UTILS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_UTILS_H

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::aster::amdgcn {
/// Get all allocas behind a value, using the following rules:
/// - If the value is not a register, it returns an empty range.
/// - If the value is an alloca, it returns the alloca.
/// - If value is a make register range it checks the inputs and if all the
/// inputs are alloca, it returns them.
/// - Otherwise, it returns failure.
FailureOr<ValueRange> getAllocasOrFailure(Value value);

/// Similar to `getAllocasOrFailure` but for a range of values.
inline LogicalResult getAllocasOrFailure(ValueRange values,
                                         SmallVectorImpl<Value> &allocas) {
  for (Value value : values) {
    FailureOr<ValueRange> result = getAllocasOrFailure(value);
    if (failed(result))
      return failure();
    llvm::append_range(allocas, *result);
  }
  return success();
}
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_UTILS_H
