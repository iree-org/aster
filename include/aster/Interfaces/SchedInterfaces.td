//===- SchedInterfaces.td - Scheduling attribute interfaces ----*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling attribute interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_SCHEDINTERFACES_TD
#define ASTER_INTERFACES_SCHEDINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SchedGraphAttrInterface
//===----------------------------------------------------------------------===//

def SchedGraphAttrInterface : AttrInterface<"SchedGraphAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that build a scheduling graph for a block.
  }];
  let methods = [
    InterfaceMethod<
      "Initialize the analyses required for scheduling",
      "::mlir::LogicalResult", "initializeAnalyses",
      (ins "::mlir::aster::SchedAnalysis &":$analysis)
    >,
    InterfaceMethod<
      "Create a scheduling graph for the given block",
      "::mlir::FailureOr<::mlir::aster::SchedGraph>", "createGraph",
      (ins "::mlir::Block *":$block, "const ::mlir::aster::SchedAnalysis &":$analysis)
    >
  ];
}

//===----------------------------------------------------------------------===//
// SchedLabelerAttrInterface
//===----------------------------------------------------------------------===//

def SchedLabelerAttrInterface : AttrInterface<"SchedLabelerAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that assign labels to nodes in a scheduling graph.
    Note, that negative labels are reserved for internal use, therefore,
    should not be used.
  }];
  let methods = [
    InterfaceMethod<
      "Get the label for a node in the scheduling graph",
      "int32_t", "getLabel",
      (ins "::mlir::Operation *":$op,
           "int32_t":$nodeId,
           "const ::mlir::aster::SchedGraph &":$graph)
    >
  ];
}

//===----------------------------------------------------------------------===//
// SchedBuilderAttrInterface
//===----------------------------------------------------------------------===//

def SchedBuilderAttrInterface : AttrInterface<"SchedBuilderAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that create a schedule (node ID order) for a graph.
    The schedule is applied via SchedGraph::applySched.
  }];
  let methods = [
    InterfaceMethod<
      "Create a schedule order for the graph, filling the given vector",
      "::mlir::LogicalResult", "createSched",
      (ins "const ::mlir::aster::SchedGraph &":$schedGraph,
           "::llvm::SmallVectorImpl<int32_t> &":$sched)
    >
  ];
}

//===----------------------------------------------------------------------===//
// SchedAttrInterface
//===----------------------------------------------------------------------===//

def SchedAttrInterface : AttrInterface<"SchedAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that aggregate scheduling components: graph
    builder, labeler, and schedule builder. Provides a match method to
    determine if the schedule applies to a block.
  }];
  let methods = [
    InterfaceMethod<
      "Get the graph builder attribute",
      "::mlir::aster::SchedGraphAttrInterface", "getGraph"
    >,
    InterfaceMethod<
      "Get the graph labeler attribute",
      "::mlir::aster::SchedLabelerAttrInterface", "getLabeler"
    >,
    InterfaceMethod<
      "Get the scheduler attribute",
      "::mlir::aster::SchedBuilderAttrInterface", "getScheduler"
    >,
    InterfaceMethod<
      "Check if this schedule should match the given block",
      "::mlir::LogicalResult", "match",
      (ins "::mlir::Block *":$block)
    >
  ];
}

#endif // ASTER_INTERFACES_SCHEDINTERFACES_TD
