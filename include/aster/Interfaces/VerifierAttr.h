//===- VerifierAttr.h - Verifier attribute interface ------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the VerifierAttr interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_VERIFIERATTR_H
#define ASTER_INTERFACES_VERIFIERATTR_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include <optional>

namespace mlir::aster {
struct VerifierRunner;
/// Scope for verifier execution.
enum class VerifierScope { RunOnRoot, RunOnNested };

/// The strictness level of the verifier.
enum class VerifierStrictness {
  Lax,
  Strict,
  Pedantic,
};

/// State provided to attribute verifiers.
struct VerifierState {
  VerifierState(Operation *root,
                std::optional<AnalysisManager> am = std::nullopt,
                VerifierStrictness strictness = VerifierStrictness::Strict)
      : root(root), am(am), strictness(strictness) {
    assert(root && "Root operation must not be null");
  }

  /// Get the root operation.
  Operation *getOp() { return root; }

  /// Get the location of the root operation.
  Location getLoc() const { return root->getLoc(); }

  /// Get an analysis from the AnalysisManager.
  template <typename AnalysisT>
  AnalysisT &getAnalysis() const {
    assert(am.has_value() && "AnalysisManager must not be valid");
    return am->getAnalysis<AnalysisT>();
  }

  /// Get an analysis from the AnalysisManager.
  template <typename AnalysisT>
  AnalysisT &getCachedAnalysis() const {
    assert(am.has_value() && "AnalysisManager must not be valid");
    return am->getCachedAnalysis<AnalysisT>();
  }

  /// Check if an AnalysisManager is available.
  bool hasAnalysisManager() const { return am.has_value(); }

  /// Check if the given operation is the root operation.
  bool isRoot(Operation *op) const { return op == root; }

  /// Get the strictness level of the verifier.
  VerifierStrictness getStrictness() const { return strictness; }

private:
  friend struct VerifierRunner;
  Operation *root;
  std::optional<AnalysisManager> am;
  VerifierStrictness strictness;
};
} // namespace mlir::aster

#include "aster/Interfaces/VerifierAttr.h.inc"

namespace mlir::aster {
/// Runner to execute verifiers on operations.
struct VerifierRunner {
  VerifierRunner(VerifierState &state)
      : context(state.root->getContext()), state(state) {}
  /// Get the verifier state.
  VerifierState &getState() { return state; }

  /// Run all verifiers on the given operation.
  LogicalResult run();

  /// Add a verifier to the runner.
  VerifierRunner &addVerifier(VerifierAttrInterface verifier) {
    assert(verifier && "Verifier must not be null");
    // TODO: Don't give access to dependent verifiers directly.
    verifier.getDependentVerifiers(verifiers);
    verifiers.push_back(verifier);
    return *this;
  }

  template <typename VerifierAttrTy, typename... Args>
  VerifierRunner &addVerifier(Args &&...args) {
    return addVerifier(VerifierAttrTy::get(std::forward<Args>(args)...));
  }

  /// Add multiple verifiers to the runner. This method includes the MLIRContext
  /// as the first argument to the verifier constructors.
  template <typename VerifierAttrTy, typename... VerifierAttrTys,
            typename... Args>
  VerifierRunner &addVerifiers(Args &&...args) {
    addVerifier(VerifierAttrTy::get(context, std::forward<Args>(args)...));
    if constexpr (sizeof...(VerifierAttrTys) > 0)
      addVerifiers<VerifierAttrTys...>(std::forward<Args>(args)...);
    return *this;
  }

private:
  MLIRContext *context;
  VerifierState &state;
  SmallVector<VerifierAttrInterface> verifiers;
};

/// Run the given verifiers on the operation.
template <typename... VerifierAttrTys, typename... Args>
LogicalResult
runVerifiersOnOp(Operation *op,
                 std::optional<AnalysisManager> am = std::nullopt,
                 VerifierStrictness strictness = VerifierStrictness::Strict,
                 Args &&...args) {
  VerifierState state(op, am, strictness);
  VerifierRunner runner(state);
  runner.template addVerifiers<VerifierAttrTys...>(std::forward<Args>(args)...);
  return runner.run();
}
} // namespace mlir::aster

#endif // ASTER_INTERFACES_VERIFIERATTR_H
