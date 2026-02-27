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
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IntEqClasses.h"

#include <optional>

namespace mlir {
class Operation;
class DataFlowSolver;
namespace aster {
namespace amdgcn {
/// Run register dead code elimination on the given operation using the
/// provided liveness solver. This function expects the liveness analysis to be
/// run before calling this function.
void registerDCE(Operation *op, DataFlowSolver &solver);

/// Class holding coalescing information for register coloring.
struct CoalescingInfo {
  using NodeID = RegisterInterferenceGraph::NodeID;

  /// Optimize the register interference graph and return equivalence classes
  /// (e.g. for coalescing). Returns std::nullopt if optimization does not
  /// apply or fails. The dataflow solver is expected to be loaded with the
  /// reaching definitions analysis tracking only loads.
  static std::optional<CoalescingInfo>
  optimizeGraph(Operation *op, RegisterInterferenceGraph &graph,
                DataFlowSolver &solver);

  /// Get the range information for a node ID. For any node returns the leader
  /// node ID, and the range constraint.
  std::pair<NodeID, RangeConstraint *>
  getRangeInfo(RegisterInterferenceGraph &graph, NodeID nodeId) {
    return graph.getRangeInfo(nodeClasses.findLeader(nodeId));
  }

  /// Get the leader of the equivalence class for the given node ID.
  NodeID getLeader(NodeID nodeId) const {
    return nodeClasses.findLeader(nodeId);
  }

  /// Equivalence classes for coalescing.
  llvm::EquivalenceClasses<int32_t> eqClasses;

private:
  /// This contains the same equivalence classes as eqClasses, but it has the
  /// guarantee that the leader of each class is the smallest member.
  llvm::IntEqClasses nodeClasses;
};
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
