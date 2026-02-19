//===- CFG.h - Control flow graph utilities ---------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_CFG_H
#define ASTER_IR_CFG_H

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::aster {
/// This class represents a branch point in the control flow graph.
class BranchPoint {
public:
  BranchPoint(RegionBranchOpInterface op, RegionBranchPoint point,
              RegionSuccessor successor)
      : operands(op.getSuccessorOperands(point, successor)) {
    this->point =
        point.isParent() ? op : point.getTerminatorPredecessorOrNull();
  }

  BranchPoint(BranchOpInterface op, int64_t index) {
    SuccessorOperands succOperands = op.getSuccessorOperands(index);
    producedOperandCount = succOperands.getProducedOperandCount();
    operands = succOperands.getForwardedOperands();
    point = op;
  }

  /// Constructor for a function entry point.
  BranchPoint(FunctionOpInterface funcOp) { (void)funcOp; }

  /// Get the operands forwarded to the branch point.
  ValueRange getOperands() const { return operands; }

  /// Get the produced operand count.
  int64_t getProducedOperandCount() const { return producedOperandCount; }

  int64_t getNumOperands() const {
    return getProducedOperandCount() + operands.size();
  }

  /// Get the point.
  Operation *getPoint() const { return point; }

  /// Check if the branch point is the entry point.
  bool isEntryPoint() const { return point == nullptr; }

private:
  int64_t producedOperandCount = 0;
  ValueRange operands;
  Operation *point = nullptr;
};

/// This class represents a successor in the control flow graph.
class Successor {
public:
  Successor(RegionBranchOpInterface brOp, RegionSuccessor succ) {
    assert(brOp && "Region branch operation must not be null");
    inputs = brOp.getSuccessorInputs(succ);
    if (succ.isParent()) {
      target = brOp;
    } else {
      Region *region = succ.getSuccessor();
      assert(!region->empty() && "Region must not be empty");
      target = &region->front();
    }
  }
  Successor(Block *block) {
    assert(block && "Block must not be null");
    inputs = block->getArguments();
    target = block;
  }

  /// Get the inputs.
  ValueRange getInputs() const { return inputs; }

  /// Get the target.
  template <typename T>
  T getTarget() const {
    return dyn_cast<T>(target);
  }

  /// Check if the successor is a block.
  bool isBlock() const { return isa<Block *>(target); }

  /// Get the target as an opaque pointer.
  void *getOpaqueTarget() const { return target.getOpaqueValue(); }

private:
  ValueRange inputs;
  llvm::PointerUnion<Operation *, Block *> target;
};

/// CRTP base for walking the control flow graph in pre-order and depth-first
/// order.
template <typename Derived>
class CFGWalker {
private:
  /// The set of visited branch points.
  DenseSet<std::pair<void *, void *>> visited;

  Derived &getDerived() { return static_cast<Derived &>(*this); }

public:
  /// Walk a region branch operation.
  LogicalResult walk(RegionBranchOpInterface regionBranchOp) {
    SmallVector<RegionSuccessor> regions;
    regionBranchOp.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor successor : regions) {
      if (failed(getDerived().handleBranch(
              BranchPoint(regionBranchOp, RegionBranchPoint::parent(),
                          successor),
              Successor(regionBranchOp, successor))))
        return failure();
    }
    return success();
  }

  /// Walk a function operation.
  LogicalResult walk(FunctionOpInterface funcOp) {
    Region &region = funcOp.getFunctionBody();
    if (region.empty())
      return success();
    return getDerived().handleBranch(BranchPoint(funcOp),
                                     Successor(&region.front()));
  }

  /// Walk a block.
  LogicalResult walk(Block *block) {
    if (!shouldVisit(block, block))
      return success();
    for (Operation &op : *block) {
      if (failed(getDerived().visitOp(&op)))
        return failure();
      if (auto brOp = dyn_cast<RegionBranchOpInterface>(&op))
        if (failed(getDerived().walk(brOp)))
          return failure();
    }

    // Handle the terminator operation.
    return getDerived().handleTerminator(block->getTerminator());
  }

protected:
  /// Check if the branch point should be visited.
  bool shouldVisit(void *src, void *tgt) {
    return visited.insert({src, tgt}).second;
  }

  /// Handle a branch in the control flow graph.
  LogicalResult handleBranch(const BranchPoint &branchPoint,
                             const Successor &successor) {
    if (!shouldVisit(branchPoint.getPoint(), successor.getOpaqueTarget()))
      return success();

    // Visit the control flow point.
    if (failed(getDerived().visitControlFlowEdge(branchPoint, successor)))
      return failure();

    // Early exit if the successor is not a block.
    if (!successor.isBlock())
      return success();

    // Walk the block.
    return getDerived().walk(successor.getTarget<Block *>());
  }

  /// Handle a terminator operation.
  LogicalResult handleTerminator(Operation *op) {
    // Handle the region branch terminator operation.
    if (auto brOp = dyn_cast<RegionBranchTerminatorOpInterface>(op))
      return getDerived().handleRegionTerminator(brOp);

    // Handle the branch operation.
    if (auto brOp = dyn_cast<BranchOpInterface>(op))
      return getDerived().handleBlockTerminator(brOp);
    return success();
  }

  /// Handle a region terminator operation.
  LogicalResult handleRegionTerminator(RegionBranchTerminatorOpInterface brOp) {
    // Handle the function return operation, and return. We need this special
    // handling because ControlFlow upstream is somewhat broken.
    if (auto funcOp = dyn_cast<FunctionOpInterface>(brOp->getParentOp())) {
      return getDerived().visitFunctionReturn(funcOp, brOp);
    }

    SmallVector<RegionSuccessor> regions;
    brOp.getSuccessorRegions({}, regions);
    auto branchOp = cast<RegionBranchOpInterface>(brOp->getParentOp());
    for (RegionSuccessor successor : regions) {
      if (failed(getDerived().handleBranch(
              BranchPoint(branchOp, RegionBranchPoint(brOp), successor),
              Successor(branchOp, successor))))
        return failure();
    }
    return success();
  }

  /// Handle a block terminator operation.
  LogicalResult handleBlockTerminator(BranchOpInterface brOp) {
    SuccessorRange successors = brOp->getSuccessors();
    for (auto [index, successor] : llvm::enumerate(successors))
      if (failed(getDerived().handleBranch(BranchPoint(brOp, index),
                                           Successor(successor))))
        return failure();
    return success();
  }

  /// Visit a function return.
  LogicalResult
  visitFunctionReturn(FunctionOpInterface funcOp,
                      RegionBranchTerminatorOpInterface returnOp) {
    (void)funcOp;
    (void)returnOp;
    return success();
  }

  /// Visit a control flow point.
  LogicalResult visitControlFlowEdge(const BranchPoint &branchPoint,
                                     const Successor &successor) {
    (void)branchPoint;
    (void)successor;
    return success();
  }

  /// Visit an operation.
  LogicalResult visitOp(Operation *op) {
    (void)op;
    return success();
  }
};
} // namespace mlir::aster

#endif // ASTER_IR_CFG_H
