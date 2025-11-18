//===- RegisterConstraints.h - Register constraints -------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_REGISTERCONSTRAINTS_H
#define ASTER_ANALYSIS_REGISTERCONSTRAINTS_H

#include "mlir/IR/Value.h"

namespace mlir {
class Attribute;
class Operation;
} // namespace mlir

namespace mlir::aster {

/// Analysis class for tracking register constraints on values.
class RegisterConstraints {
public:
  /// Create a RegisterConstraints analysis for the given operation.
  /// Returns failure if the constraints are not satisfiable.
  static FailureOr<RegisterConstraints> create(Operation *op);

  /// Get the register constraint attribute for the given value.
  /// Returns nullptr if no constraint exists for the value.
  Attribute getConstraint(Value value) const {
    return constraints.lookup_or(value, nullptr);
  }

private:
  RegisterConstraints() = default;

  /// Map from values to their register constraint attributes.
  DenseMap<Value, Attribute> constraints;
};
} // namespace mlir::aster

#endif // ASTER_ANALYSIS_REGISTERCONSTRAINTS_H
