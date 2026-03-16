//===- ApplySched.cpp - Apply scheduling attributes -----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/SchedInterfaces.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "aster-sched"

namespace mlir::aster {
#define GEN_PASS_DEF_APPLYSCHED
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ApplySchedPass
//===----------------------------------------------------------------------===//

struct ApplySchedPass : public aster::impl::ApplySchedBase<ApplySchedPass> {
  using ApplySchedBase::ApplySchedBase;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// SchedInfo
//===----------------------------------------------------------------------===//
/// SchedInfo is a tuple of the scope operation, the name of the schedule, and
/// the schedule attribute.
struct SchedInfo {
  Operation *scope = nullptr;
  StringAttr name;
  SchedAttrInterface schedAttr;
  int64_t id;
};
} // namespace

/// Collect all SchedAttrInterface attributes by walking ops in preorder.
static void collectSchedAttrs(Operation *root,
                              SmallVectorImpl<SchedInfo> &schedInfos,
                              DenseMap<StringAttr, int32_t> &schedsNames) {
  int64_t id = 0;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumRegions() == 0)
      return;

    // Traverse the discardable attributes of the operation.
    for (NamedAttribute namedAttr : op->getDiscardableAttrs()) {
      auto schedAttr = dyn_cast<SchedAttrInterface>(namedAttr.getValue());
      if (!schedAttr)
        continue;

      // Check if we are interested in this schedule.
      auto it = schedsNames.find(namedAttr.getName());
      if (it == schedsNames.end())
        continue;

      // Increment the count of this schedule name.
      it->second++;

      // Add the schedule info to the list.
      schedInfos.push_back({op, namedAttr.getName(), schedAttr, id++});
    }
  });
}

/// Verify that all names in scheds appear in the map.
static LogicalResult
verifySchedNames(Operation *root, const DenseMap<StringAttr, int32_t> &map) {
  for (const auto &[key, count] : map) {
    if (count != 0)
      continue;
    return root->emitError()
           << "schedule '" << key.strref()
           << "' not found. Expected a SchedAttrInterface attribute "
              "with this name in a discardable attribute dictionary.";
  }
  return success();
}

/// Apply a single schedule to all the immediately nested blocks.
static LogicalResult applySched(SchedInfo sched, SchedAnalysis &analysis,
                                IRRewriter &rewriter) {
  Operation *op = sched.scope;
  StringAttr name = sched.name;
  SchedAttrInterface schedAttr = sched.schedAttr;

  LDBG() << "Applying schedule '" << name.strref() << "' " << schedAttr
         << " to operation: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      // Check if the schedule matches the block.
      if (failed(schedAttr.match(&block)))
        continue;

      // Get the graph builder attribute.
      SchedGraphAttrInterface graphBuilder = schedAttr.getGraph();
      assert(graphBuilder && "expected a valid graph builder");

      // Create the scheduling graph.
      FailureOr<SchedGraph> graph = graphBuilder.createGraph(&block, analysis);
      if (failed(graph))
        return op->emitError() << "failed to create graph for '" << name << "'";

      LDBG_OS([&](llvm::raw_ostream &os) { graph->print(os); });

      // Get the labeler attribute.
      SchedLabelerAttrInterface labeler = schedAttr.getLabeler();
      assert(labeler && "expected a valid labeler");

      // Label the nodes in the graph.
      for (int64_t i = 0, e = static_cast<int64_t>(graph->getOps().size());
           i < e; ++i) {
        Operation *nodeOp = graph->getOp(i);
        int32_t label =
            labeler.getLabel(nodeOp, static_cast<int32_t>(i), *graph);
        if (label < 0)
          continue;
        graph->setLabel(i, label);
      }

      LDBG() << "Labels: " << llvm::interleaved_array(graph->getLabels());

      // Create the schedule and apply it to the block.
      SmallVector<int32_t> order;
      rewriter.setInsertionPointToStart(&block);
      if (failed(schedAttr.getScheduler().createSched(*graph, order)))
        return op->emitError() << "failed to create schedule '" << name << "'";

      LDBG() << "Schedule: " << llvm::interleaved_array(order);
      SchedGraph::applySched(*graph, rewriter, order);
    }
  }

  sched.scope->removeDiscardableAttr(name);
  return success();
}

//===----------------------------------------------------------------------===//
// ApplySchedPass
//===----------------------------------------------------------------------===//

void ApplySchedPass::runOnOperation() {
  Operation *root = getOperation();

  // Error out if there are no schedules to apply.
  if (scheds.empty()) {
    root->emitError() << "no schedules to apply";
    return signalPassFailure();
  }

  IRRewriter rewriter(root->getContext());
  SmallVector<SchedInfo> schedInfos;

  // Initialize the counts for the schedules of interest.
  DenseMap<StringAttr, int32_t> schedsCounts;
  DenseMap<StringAttr, int64_t> schedsToId;
  for (const auto &[id, name] : llvm::enumerate(scheds)) {
    StringAttr key = StringAttr::get(root->getContext(), name);
    schedsCounts[key] = 0;
    schedsToId[key] = id;
  }

  // Collect the schedules of interest.
  collectSchedAttrs(root, schedInfos, schedsCounts);

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Requested schedules: " << llvm::interleaved_array(scheds);
  });

  // Verify that all schedules of interest were found.
  // if (failed(verifySchedNames(root, schedsCounts)))
  //   return signalPassFailure();

  // Sort the schedules by order of appearance in the scheds list and by
  // appearance in the walk.
  llvm::sort(schedInfos, [&schedsToId](const SchedInfo &a, const SchedInfo &b) {
    return std::make_pair(schedsToId[a.name], a.id) <
           std::make_pair(schedsToId[b.name], b.id);
  });

  if (schedInfos.empty())
    return;

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Applying schedules:\n";
    llvm::interleave(
        schedInfos, os,
        [&](const SchedInfo &schedInfo) {
          os << "  Name: `" << schedInfo.name.strref()
             << "`\n    Attribute: " << schedInfo.schedAttr;
          os << "\n    Scope: "
             << OpWithFlags(schedInfo.scope, OpPrintingFlags().skipRegions());
        },
        "\n");
  });

  // Configure the analyses required for scheduling.
  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  AnalysisManager analysisManager = getAnalysisManager();
  SchedAnalysis analysis(root, solver, domInfo, analysisManager);

  // Initialize the analyses required for scheduling.
  for (SchedInfo &schedInfo : schedInfos) {
    if (failed(schedInfo.schedAttr.getGraph().initializeAnalyses(analysis))) {
      root->emitError() << "failed to initialize analyses for schedule '"
                        << schedInfo.name.strref()
                        << "': " << schedInfo.schedAttr.getGraph();
      return signalPassFailure();
    }
  }

  if (analysis.shouldRunDataflowAnalyses()) {
    if (failed(solver.initializeAndRun(root))) {
      root->emitError() << "failed to run dataflow analyses";
      return signalPassFailure();
    }
  }

  // Apply the schedules in order.
  for (const SchedInfo &schedInfo : schedInfos) {
    if (failed(applySched(schedInfo, analysis, rewriter)))
      return signalPassFailure();
  }
}
