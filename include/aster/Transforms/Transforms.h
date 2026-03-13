//===- Transforms.h - Non-pass transformation APIs ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone transformations that are called from passes but are not passes
// themselves. Each has a corresponding test pass in test/lib/Pass/ for unit
// testing.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_TRANSFORMS_H
#define ASTER_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::aster {

/// Factorize an affine expression to minimize the number of multiplications.
///
/// Expands the expression into a sum-of-monomials polynomial, greedily
/// extracts the most-frequent variable as a common factor, then post-processes
/// to extract any remaining shared multiplicative factors across addends.
/// The two phases repeat until fixpoint.  Mod/FloorDiv/CeilDiv sub-expressions
/// are recursively factorized but treated as atomic during polynomial
/// expansion.
AffineExpr factorizeAffineExpr(AffineExpr expr);

/// Prepare LDS buffers for multi-buffering in all scf.for loops under `op`.
///
/// For each loop containing alloc_lds/dealloc_lds pairs with sched.stage
/// annotations, hoists N copies of each buffer before the loop and replaces
/// in-loop LDS offset usage with rotating iter_args.
///
/// Returns failure if any loop transformation fails.
LogicalResult prepareLDSMultibuffers(Operation *op);

} // namespace mlir::aster

#endif // ASTER_TRANSFORMS_TRANSFORMS_H
