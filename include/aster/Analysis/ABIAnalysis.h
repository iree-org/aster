//===- ABIAnalysis.h - ABI analysis ------------------------------*- C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_ABIANALYSIS_H
#define ASTER_ANALYSIS_ABIANALYSIS_H

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/Support/TypeSize.h"

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// ABIAnalysis
//===----------------------------------------------------------------------===//
/// Analysis providing ABI-related information such as data layout and
/// thread-uniformity of values.
struct ABIAnalysis {
  ABIAnalysis(Operation *op, AnalysisManager &am)
      : dataLayoutAnalysis(am.getAnalysis<DataLayoutAnalysis>()) {
    dataLayout = &dataLayoutAnalysis.getAtOrAbove(op);
    mlir::dataflow::loadBaselineAnalyses(solver);
    solver.load<aster::dataflow::ThreadUniformAnalysis>();
    LogicalResult result = solver.initializeAndRun(op);
    assert(succeeded(result) && "Failed to run ABIAnalysis");
    (void)result;
  }

  /// Check if the given value is known to be thread-uniform.
  std::optional<bool> isThreadUniform(Value value) const {
    if (!value)
      return std::nullopt;
    auto state = solver.lookupState<dataflow::ThreadUniformLattice>(value);
    if (!state)
      return std::nullopt;
    return state->getValue().isUniform();
  }

  /// Get the size of the given type according to the data layout.
  llvm::TypeSize getTypeSize(Type type) const {
    return dataLayout->getTypeSize(type);
  }
  llvm::TypeSize getTypeSizeInBits(Type type) const {
    return dataLayout->getTypeSizeInBits(type);
  }
  /// Get the ABI alignment of the given type according to the data layout.
  uint64_t getAlignment(Type type) const {
    return dataLayout->getTypeABIAlignment(type);
  }

  /// Check whether the analysis is invalidated.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<DataLayoutAnalysis>() ||
           !pa.isPreserved<ABIAnalysis>();
  }

private:
  /// The data-flow solver.
  DataFlowSolver solver;
  /// The data layout analysis.
  DataLayoutAnalysis &dataLayoutAnalysis;
  const DataLayout *dataLayout;
};
} // end namespace mlir::aster

#endif // ASTER_ANALYSIS_ABIAnalysis_H
