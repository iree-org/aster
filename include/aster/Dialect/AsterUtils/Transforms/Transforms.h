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
} // namespace mlir

namespace mlir::aster {
namespace aster_utils {

/// Wraps all function calls in the given operation with execute_region
/// operations. Calls that are already inside an execute_region are skipped.
void wrapCallsWithExecuteRegion(Operation *op);

/// Inlines all execute_region operations in the given operation, replacing
/// them with their body contents.
void inlineExecuteRegions(Operation *op);

} // namespace aster_utils
} // namespace mlir::aster

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_TRANSFORMS_H
