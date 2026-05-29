//===- Transforms.h - AMDGCN Transform Utilities -----------------*- C++
//-*-===//
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

#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Support/Graph.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class DataFlowSolver;
namespace aster {
namespace amdgcn {
/// Run register dead code elimination on the given operation using the
/// provided liveness solver. This function expects the liveness analysis to be
/// run before calling this function.
void registerDCE(Operation *op, DataFlowSolver &solver);

/// The interference graph remapped through equivalence classes, with
/// coalescing information for register coloring.
struct CoalescingInfo {
  /// Optimize the register interference graph by coalescing non-interfering
  /// registers connected by move operations, and return the resulting quotient
  /// graph. Returns failure if move analysis encounters an unexpected register
  /// form.
  static FailureOr<CoalescingInfo>
  optimizeGraph(Operation *op, RegisterInterferenceGraph &graph);

  /// The interference graph remapped through equivalence classes (undirected).
  Graph graph{/*directed=*/false};
  /// Representative value for each quotient node.
  SmallVector<Value> values;
  /// Range constraint for each quotient node (nullptr if singleton).
  SmallVector<RangeConstraint *> constraints;
  /// Flat storage of original node IDs grouped by quotient class, in
  /// ascending order within each class. Class qid occupies
  /// memberData[memberOffsets[qid] .. memberOffsets[qid+1]).
  /// memberData[memberOffsets[qid]] is the minimum (representative) of qid.
  SmallVector<int32_t> memberData;
  SmallVector<int32_t> memberOffsets;
  /// nodeClass[origId] is the quotient class ID of original node origId.
  SmallVector<int32_t> nodeClass;
};
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
