//===- SchedOps.cpp - Sched dialect operations ------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Sched/IR/SchedOps.h"
#include "aster/Dialect/Sched/IR/SchedDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::sched;

//===----------------------------------------------------------------------===//
// Sched Inliner Interface
//===----------------------------------------------------------------------===//

namespace {
struct SchedInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Always allow inlining of Sched operations.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Always allow inlining of Sched operations into regions.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &mapping) const final {
    return true;
  }

  /// Always allow inlining of regions.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Sched dialect
//===----------------------------------------------------------------------===//

void SchedDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/Sched/IR/SchedOps.cpp.inc"
      >();
  addInterfaces<SchedInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// UnitOp
//===----------------------------------------------------------------------===//

void UnitOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // When branching from the parent op, enter the body region.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  // The body region branches back to the parent op.
  regions.push_back(RegionSuccessor::parent());
}

void UnitOp::getSuccessorRegions(Region &region,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // The body region always branches back to the parent op.
  regions.push_back(RegionSuccessor::parent());
}

ValueRange UnitOp::getSuccessorInputs(RegionSuccessor successor) {
  // When returning to the parent, the successor inputs are the op's results.
  if (successor.isParent())
    return ValueRange(getOperation()->getResults());
  // The body region is isolated; its block arguments are not part of the
  // region branch interface flow.
  return ValueRange();
}

LogicalResult UnitOp::verify() {
  Block &block = getBody().front();

  // Verify block args correspond to inputs.
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("body block argument count (")
           << block.getNumArguments() << ") must match the number of inputs ("
           << getInputs().size() << ")";
  for (auto [arg, input] : llvm::zip(block.getArguments(), getInputs()))
    if (arg.getType() != input.getType())
      return emitOpError("body block argument type ")
             << arg.getType() << " does not match input type "
             << input.getType();

  // Verify terminator.
  if (!isa<YieldOp>(block.getTerminator()))
    return emitOpError("body must be terminated by sched.yield");
  YieldOp yield = cast<YieldOp>(block.getTerminator());
  if (yield.getValues().getTypes() != getResults().getTypes())
    return emitOpError("sched.yield types must match op result types");
  return success();
}

//===----------------------------------------------------------------------===//
// LoopResourceOp
//===----------------------------------------------------------------------===//

LogicalResult LoopResourceOp::verify() {
  // Must be inside a loop-like op.
  if (!getOperation()->getParentOfType<LoopLikeOpInterface>())
    return emitOpError("must be inside a loop-like op");

  // allocate must yield at least one value.
  Block &allocBlock = getAllocate().front();
  if (!isa<YieldOp>(allocBlock.getTerminator()))
    return emitOpError("allocate region must be terminated by sched.yield");
  YieldOp allocYield = cast<YieldOp>(allocBlock.getTerminator());
  if (allocYield.getValues().empty())
    return emitOpError("allocate region must yield at least one value");

  // forward must yield values matching the op's results.
  Block &fwdBlock = getForward().front();
  if (!isa<YieldOp>(fwdBlock.getTerminator()))
    return emitOpError("forward region must be terminated by sched.yield");
  YieldOp fwdYield = cast<YieldOp>(fwdBlock.getTerminator());
  if (fwdYield.getValues().getTypes() != getResults().getTypes())
    return emitOpError("forward region yield types must match op result types");

  // Check optional region terminators.
  if (!getDeallocate().empty()) {
    Block &deallBlock = getDeallocate().front();
    if (!isa<YieldOp>(deallBlock.getTerminator()))
      return emitOpError("deallocate region must be terminated by sched.yield");
  }
  if (!getFence().empty()) {
    Block &fenceBlock = getFence().front();
    if (!isa<YieldOp>(fenceBlock.getTerminator()))
      return emitOpError("fence region must be terminated by sched.yield");
  }

  return success();
}

ParseResult LoopResourceOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional result types: -> type, type, ...
  if (succeeded(parser.parseOptionalArrow())) {
    SmallVector<Type> resultTypes;
    if (parser.parseTypeList(resultTypes))
      return failure();
    result.addTypes(resultTypes);
  }

  // Parse allocate region (no block args at op-level; they're inside the {}).
  Region *allocate = result.addRegion();
  if (parser.parseRegion(*allocate))
    return failure();

  // Parse optional deallocate region.
  Region *deallocate = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("deallocate")))
    if (parser.parseRegion(*deallocate))
      return failure();

  // Parse (required) forward region.
  Region *forward = result.addRegion();
  if (parser.parseKeyword("forward") || parser.parseRegion(*forward))
    return failure();

  // Parse optional fence region.
  Region *fence = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("fence")))
    if (parser.parseRegion(*fence))
      return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void LoopResourceOp::print(OpAsmPrinter &p) {
  // Print result types.
  if (!getResults().empty()) {
    p << " ->";
    llvm::interleaveComma(getResults().getTypes(), p.getStream());
  }

  // Print allocate region (block args are printed inside the region body).
  p << " ";
  p.printRegion(getAllocate(), /*printEntryBlockArgs=*/false);

  // Print optional deallocate region.
  if (!getDeallocate().empty()) {
    p << " deallocate";
    p.printRegion(getDeallocate(), /*printEntryBlockArgs=*/true);
  }

  // Print forward region.
  p << " forward";
  p.printRegion(getForward(), /*printEntryBlockArgs=*/true);

  // Print optional fence region.
  if (!getFence().empty()) {
    p << " fence";
    p.printRegion(getFence(), /*printEntryBlockArgs=*/true);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/Sched/IR/SchedOps.cpp.inc"

#include "aster/Dialect/Sched/IR/SchedDialect.cpp.inc"
