//===- Transforms.h - Common AsterUtils transforms ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_TRANSFORMS_TRANSFORMS_H
#define ASTER_DIALECT_ASTERUTILS_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class Operation;
class RewritePatternSet;
} // namespace mlir

namespace mlir::aster {
namespace aster_utils {

/// Populates the pattern set with the FromAnyOp to ub.poison conversion
/// pattern.
void populateFromAnyToPoisonPattern(RewritePatternSet &patterns);

} // namespace aster_utils
} // namespace mlir::aster

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_TRANSFORMS_H
