//===- Transforms.h - AMDGCN Transform Utilities ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
#define ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class DominanceInfo;
class Operation;
} // namespace mlir

namespace mlir::aster::amdgcn {
/// Canonicalize wait operations in the given operation. If useLICM is true,
/// LICM will be used to hoist wait operations out of loops where possible. If
/// weakDepsAsDeps is true, weak dependencies will be treated as strong
/// dependencies.
LogicalResult canonicalizeWaits(Operation *op, DominanceInfo *domInfo,
                                bool useLICM);
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
