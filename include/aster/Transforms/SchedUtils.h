//===- SchedUtils.h - Schedule utility functions and constants  -----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_SCHEDUTILS_H
#define ASTER_TRANSFORMS_SCHEDUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::aster {
// Schedule attribute name constants
constexpr StringLiteral kSchedDelayAttr = "sched.delay";
constexpr StringLiteral kSchedRateAttr = "sched.rate";
constexpr StringLiteral kSchedPermutationAttr = "sched.permutation";
constexpr StringLiteral kSchedDimsAttr = "sched.dims";
constexpr StringLiteral knownSchedAttrs[] = {
    StringLiteral(kSchedDelayAttr), StringLiteral(kSchedRateAttr),
    StringLiteral(kSchedPermutationAttr), StringLiteral(kSchedDimsAttr)};

/// Helper function to check if an operation has any of the known schedule
/// attributes. This function exhaustively checks all known schedule attribute
/// constants defined above.
inline bool hasSchedAttribute(Operation *op) {
  if (!op)
    return false;

  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    for (StringLiteral schedAttr : knownSchedAttrs) {
      if (name == schedAttr)
        return true;
    }
  }
  return false;
}
} // namespace mlir::aster

#endif // ASTER_TRANSFORMS_SCHEDUTILS_H
