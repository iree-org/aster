//===- MemoryDependenceAnalysis.h - Memory dependence analysis ---*- C++-*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Late memory dependence analysis operating on CFG IR.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H
#define ASTER_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::aster {

enum class DepKind { RAW, WAR, WAW };
struct MemDepEdge {
  Operation *producer;
  DepKind kind;
  mlir::SideEffects::Resource *resource;
};

/// Cross-block memory dependence analysis. At construction time the analysis
/// collects memory access sites, assigns alias equivalence classes, and
/// precomputes RAW/WAR/WAW dependences ending at each operation.
class MemoryDependenceAnalysis {
public:
  /// Verify the flat-CFG normal form on `root` (#amdgcn.no_scf_ops), then
  /// build alias classes and dependence summaries for the whole region.
  static FailureOr<MemoryDependenceAnalysis>
  create(Operation *root, ArrayRef<SideEffects::Resource *> resources);

  /// Precomputed dependences ending at `op` (empty if none or no memory
  /// effects on the tracked resources).
  ArrayRef<MemDepEdge> getDependences(Operation *op) const;

private:
  struct AccessSite {
    Operation *op;
    bool isWrite;
    SideEffects::Resource *resource;
    int64_t aliasClass;
  };

  MemoryDependenceAnalysis(ArrayRef<SideEffects::Resource *> resources);

  void buildAccessSites(Operation *root);
  void buildDependences(Operation *root);

  bool mayAlias(const AccessSite &a, const AccessSite &b) const;

  SmallVector<SideEffects::Resource *, 2> resources;
  SmallVector<AccessSite> accessSites;
  DenseMap<Operation *, SmallVector<int64_t, 2>> opAccessIndices;
  DenseMap<SideEffects::Resource *, int64_t> unknownAliasClass;
  int64_t nextAliasClass = 0;
  DenseMap<Operation *, SmallVector<MemDepEdge>> dependences;
  static const SmallVector<MemDepEdge> emptyDeps;
};

} // namespace mlir::aster

#endif // ASTER_ANALYSIS_MEMORYDEPENDENCEANALYSIS_H
