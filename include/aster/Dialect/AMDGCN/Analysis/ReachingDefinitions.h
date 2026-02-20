//===- ReachingDefinitions.h - Reaching definitions analysis ----*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dense forward dataflow analysis that computes reaching definitions at each
// program point. Only considers effects of InstOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_REACHINGDEFINITIONS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_REACHINGDEFINITIONS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <functional>
#include <set>
#include <utility>

namespace mlir::aster {
class SSAMap;
}

namespace mlir::aster::amdgcn {

//===----------------------------------------------------------------------===//
// Definition
//===----------------------------------------------------------------------===//

/// A reaching definition.
struct Definition {
  Value allocation;
  OpOperand *definition;

  Definition(Value allocation, OpOperand &definition)
      : Definition(allocation, &definition) {
    assert(allocation && "Allocation must be non-null");
  }

  /// Create a lower bound definition for the given allocation.
  static Definition createLowerBound(Value allocation) {
    return Definition{allocation, nullptr};
  }

  bool operator<(const Definition &other) const {
    return std::make_pair(allocation.getAsOpaquePointer(), definition) <
           std::make_pair(other.allocation.getAsOpaquePointer(),
                          other.definition);
  }

private:
  Definition(Value allocation, OpOperand *definition)
      : allocation(allocation), definition(definition) {}
};

//===----------------------------------------------------------------------===//
// ReachingDefinitionsState
//===----------------------------------------------------------------------===//

/// Lattice element: set of reaching definitions at a program point.
struct ReachingDefinitionsState : dataflow::AbstractDenseLattice {
  using DefinitionSet = std::set<Definition>;

  ReachingDefinitionsState(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor) {}

  /// Join (merge) with another state: union of definition sets.
  ChangeResult join(const ReachingDefinitionsState &other);
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    return join(static_cast<const ReachingDefinitionsState &>(lattice));
  }

  /// Remove all definitions that write to the given allocation.
  ChangeResult killDefinitions(Value allocation);

  /// Add a definition.
  ChangeResult addDefinition(Definition definition);

  /// Print the state.
  void print(raw_ostream &os) const override;

  /// Print the state with deterministic order: sort by SSA ID of allocation,
  /// dominance of definition owner, then operand number.
  void print(raw_ostream &os, const mlir::aster::SSAMap &ssaMap,
             const DominanceInfo &dominance) const;

  /// Get the set of definitions.
  const DefinitionSet &getDefinitions() const { return definitions; }

  /// Set the state to the entry state.
  ChangeResult setToEntryState() {
    if (definitions.empty())
      return ChangeResult::NoChange;
    definitions.clear();
    return ChangeResult::Change;
  }

private:
  DefinitionSet definitions;
};

//===----------------------------------------------------------------------===//
// ReachingDefinitionsAnalysis
//===----------------------------------------------------------------------===//

/// Dense forward dataflow analysis that computes, at each program point, the
/// set of DPS `outs` operands that may have last written to each allocation.
///
/// Precondition: the IR must be in post-bufferization DPS normal form (i.e.
/// instructions write exclusively through side-effecting `outs` operands with
/// no SSA results carrying register values). Such a precondition is guaranteed
/// by the ToRegisterSemantics pass.
///
/// An optional `definitionFilter` controls which operations generate
/// definitions. Filtered-out operations still kill previous definitions to
/// their `outs` allocations (the write happens regardless), but do not add
/// themselves to the reaching set. This enables queries like "which loads reach
/// this point?" while preserving correct kill semantics for intervening
/// non-load writes.
class ReachingDefinitionsAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ReachingDefinitionsState> {
  using Base = dataflow::DenseForwardDataFlowAnalysis<ReachingDefinitionsState>;

public:
  /// Verify the DPS normal form precondition on `root`, then load the analysis
  /// into `solver`. Returns failure if any InstOpInterface `outs` operand has
  /// value semantics (i.e. the IR has not been through ToRegisterSemantics).
  /// The caller is responsible for calling `solver.initializeAndRun(root)`.
  static FailureOr<ReachingDefinitionsAnalysis *>
  create(DataFlowSolver &solver, Operation *root,
         llvm::function_ref<bool(Operation *)> definitionFilter = {});

  /// Visit an operation and update the reaching definitions state.
  LogicalResult visitOperation(Operation *op,
                               const ReachingDefinitionsState &before,
                               ReachingDefinitionsState *after) override;

  /// Set the entry state.
  void setToEntryState(ReachingDefinitionsState *lattice) override;

private:
  friend class ::mlir::DataFlowSolver;
  ReachingDefinitionsAnalysis(
      DataFlowSolver &solver,
      llvm::function_ref<bool(Operation *)> definitionFilter)
      : Base(solver), definitionFilter(definitionFilter) {}

  /// A filter function that determines if a definition should be tracked. This
  /// function should return true if the definition should be tracked, and false
  /// otherwise.
  llvm::function_ref<bool(Operation *)> definitionFilter;
};

} // end namespace mlir::aster::amdgcn

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::ReachingDefinitionsState)

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_REACHINGDEFINITIONS_H
