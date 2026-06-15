//===- TestMemoryDependenceAnalysis.cpp - Test Memory Dependence Analysis ===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass that prints the RAW/WAR/WAW memory-dependence edges ending at each
// operation, for FileCheck verification of MemoryDependenceAnalysis.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-memory-dependence-analysis"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTMEMORYDEPENDENCEANALYSIS
#include "Passes.h.inc"
} // namespace mlir::aster::test

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// TestMemoryDependenceAnalysis pass
//===----------------------------------------------------------------------===//
class TestMemoryDependenceAnalysis
    : public mlir::aster::test::impl::TestMemoryDependenceAnalysisBase<
          TestMemoryDependenceAnalysis> {
public:
  using TestMemoryDependenceAnalysisBase::TestMemoryDependenceAnalysisBase;

  void runOnOperation() override {
    Operation *root = getOperation();

    llvm::outs() << "=== Memory Dependence Analysis Results ===\n";

    root->walk([&](KernelOp kernel) {
      llvm::outs() << "\nKernel: " << kernel.getSymName() << "\n";

      // Deterministic producer ordering: index ops in program order.
      llvm::DenseMap<Operation *, int> order;
      int idx = 0;
      kernel.walk([&](Operation *op) { order[op] = idx++; });

      // Cross-block analysis: verify the flat-CFG normal form, then query.
      auto mdaOr = MemoryDependenceAnalysis::create(
          kernel, {GlobalMemoryResource::get(), LDSMemoryResource::get()});
      if (failed(mdaOr))
        return;
      MemoryDependenceAnalysis &mda = *mdaOr;
      kernel.walk([&](Operation *op) {
        bool hasTestAttr = llvm::any_of(op->getAttrs(), [](NamedAttribute a) {
          return a.getName().getValue().starts_with("test.");
        });
        if (!hasTestAttr)
          return;

        llvm::outs() << "\nOperation: " << *op << "\n";
        ArrayRef<MemDepEdge> edges = mda.getDependences(op);
        printKind(edges, DepKind::RAW, "RAW", order);
        printKind(edges, DepKind::WAR, "WAR", order);
        printKind(edges, DepKind::WAW, "WAW", order);
      });
    });

    llvm::outs() << "\n=== End Analysis Results ===\n";
  }

private:
  static void printTestAttrs(Operation *op) {
    for (NamedAttribute a : op->getAttrs())
      if (a.getName().getValue().starts_with("test."))
        llvm::outs() << a.getName().getValue() << ", ";
  }

  /// Print one edge-kind line: count, then the producers' test.* attributes in
  /// program order.
  static void printKind(ArrayRef<MemDepEdge> edges, DepKind kind,
                        StringRef label,
                        const llvm::DenseMap<Operation *, int> &order) {
    SmallVector<Operation *> producers;
    for (const MemDepEdge &e : edges)
      if (e.kind == kind)
        producers.push_back(e.producer);
    llvm::sort(producers, [&](Operation *a, Operation *b) {
      return order.lookup(a) < order.lookup(b);
    });
    llvm::outs() << "\t" << label << " deps ending here: " << producers.size()
                 << ": ";
    for (Operation *p : producers)
      printTestAttrs(p);
    llvm::outs() << "\n";
  }
};
} // namespace
