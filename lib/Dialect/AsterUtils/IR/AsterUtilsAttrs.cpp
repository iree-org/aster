//===- AsterUtilsAttrs.cpp - AsterUtils attributes --------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Interfaces/SchedInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.cpp.inc"

void AsterUtilsDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aster/Dialect/AsterUtils/IR/AsterUtilsAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SSASchedulerAttr - SchedGraphAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult SSASchedulerAttr::initializeAnalyses(SchedAnalysis &) const {
  return success();
}

FailureOr<SchedGraph>
SSASchedulerAttr::createGraph(Block *block, const SchedAnalysis &) const {
  SchedGraph graph(block);
  for (Operation &op : block->getOperations()) {
    for (Value operand : op.getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getBlock() == block)
        graph.addEdge(producer, &op);
    }
  }
  graph.compress();
  return graph;
}

//===----------------------------------------------------------------------===//
// OpNameLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t OpNameLabelerAttr::getLabel(Operation *op, int32_t,
                                    const SchedGraph &) const {
  ArrayRef<StringAttr> matcher = getOpNameMatcher();
  if (matcher.empty())
    return getStage();
  StringAttr opName = op->getName().getIdentifier();
  if (!llvm::any_of(matcher,
                    [&](StringAttr nameAttr) { return nameAttr == opName; }))
    return -1;
  return getStage();
}

//===----------------------------------------------------------------------===//
// SchedListLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t SchedListLabelerAttr::getLabel(Operation *op, int32_t nodeId,
                                       const SchedGraph &graph) const {
  for (SchedLabelerAttrInterface labeler : getLabelers()) {
    int32_t label = labeler.getLabel(op, nodeId, graph);
    if (label >= 0)
      return label;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// SchedStageLabelerAttr - SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

int32_t SchedStageLabelerAttr::getLabel(Operation *op, int32_t,
                                        const SchedGraph &) const {
  auto attr =
      dyn_cast_or_null<IntegerAttr>(op->getDiscardableAttr("sched.stage"));
  if (!attr)
    return std::numeric_limits<int32_t>::max();
  int64_t value = attr.getValue().getSExtValue();
  if (value < 0 || value > std::numeric_limits<int32_t>::max())
    return std::numeric_limits<int32_t>::max();
  return static_cast<int32_t>(value);
}

//===----------------------------------------------------------------------===//
// StageTopoSortSchedAttr - SchedBuilderAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult
StageTopoSortSchedAttr::createSched(const SchedGraph &schedGraph,
                                    SmallVectorImpl<int32_t> &sched) const {
  if (!schedGraph.isCompressed())
    return failure();

  // Use topologicalSched with a stage-aware selection function.
  // Given ready nodes, pick the one with smallest stage label; ties by node ID.
  auto schedFn = [&](ArrayRef<int32_t> ready) -> int32_t {
    int32_t bestIdx = 0;
    int32_t bestLabel = schedGraph.getLabel(ready[0]);
    int32_t bestNode = ready[0];
    for (size_t i = 1; i < ready.size(); ++i) {
      int32_t label = schedGraph.getLabel(ready[i]);
      if (label < bestLabel || (label == bestLabel && ready[i] < bestNode)) {
        bestIdx = i;
        bestLabel = label;
        bestNode = ready[i];
      }
    }
    return bestIdx;
  };

  return schedGraph.topologicalSched(schedFn, sched);
}

//===----------------------------------------------------------------------===//
// GenericSchedulerAttr - SchedAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult GenericSchedulerAttr::match(Block *) const { return success(); }
