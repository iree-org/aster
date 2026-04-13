//===- LowerSRegBlockArgs.cpp - Tunnel SREG block args through SGPR ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers AMDGCN special-register block arguments to SGPR carriers.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/CFG.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LOWERSREGBLOCKARGS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Returns true if `arg` is a special register with value semantics.
static bool isSpecialReg(Type type) {
  auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(type);
  if (!regTy || !regTy.hasValueSemantics())
    return false;
  RegisterKind k = regTy.getRegisterKind();
  if (k == RegisterKind::Unknown || k == RegisterKind::AGPR ||
      k == RegisterKind::SGPR || k == RegisterKind::VGPR)
    return false;
  return true;
}

namespace {
//===----------------------------------------------------------------------===//
// LowerSRegBlockArgs
//===----------------------------------------------------------------------===//
struct LowerSRegBlockArgs
    : public amdgcn::impl::LowerSRegBlockArgsBase<LowerSRegBlockArgs> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// TransformImpl
//===----------------------------------------------------------------------===//

/// An edge in the control flow graph that needs to be lowered.
struct ForwardingEdge {
  BranchOpInterface brOp;
  BlockArgument destArg;
  int32_t successorIndex;
};

/// Collects SREG phi edges, rewrites predecessors through SGPR carriers,
/// then materializes SREG values in successors.
class TransformImpl {
public:
  explicit TransformImpl(FunctionOpInterface funcOp)
      : funcOp(funcOp), rewriter(funcOp.getContext()) {}

  /// Walks the function body, rewrites all qualifying block arguments, or
  /// returns failure.
  LogicalResult run();

  /// Inserts alloca + `lsir.copy` before the branch and patches one successor
  /// operand.
  void handleBranchOpEdge(const ForwardingEdge &edge,
                          RegisterTypeInterface sgprTy);

  /// Handles the given block by changing the block argument types and adding
  /// copies for the SREGs.
  void handleBlock(Block *bb, ArrayRef<Type> newArgTys,
                   ArrayRef<Location> bbArgLocs);

  FunctionOpInterface funcOp;
  IRRewriter rewriter;
  SmallVector<ForwardingEdge> forwardingEdges;
};

/// CFG walker that appends to `forwardingEdges`.
struct Collector : CFGWalker<Collector> {
  TransformImpl *impl = nullptr;
  LogicalResult visitControlFlowEdge(const BranchPoint &branchPoint,
                                     const Successor &successor);
};
} // namespace

LogicalResult Collector::visitControlFlowEdge(const BranchPoint &branchPoint,
                                              const Successor &successor) {
  if (branchPoint.isEntryPoint() || !successor.isBlock())
    return success();

  // Bail out if the branch point has produced operands.
  if (branchPoint.getProducedOperandCount() > 0)
    return branchPoint.getPoint()->emitError()
           << "expected a branch point with no produced operands";

  // Get the branch point operands and successor block arguments.
  ValueRange cfOperands = branchPoint.getOperands();
  ValueRange succVars = successor.getInputs();

  // Get the forwarding edges. We only care about block arguments with special
  // register value semantics. zip_equal is used to assert that operand and
  // argument counts match.
  BranchOpInterface brOp = dyn_cast<BranchOpInterface>(branchPoint.getPoint());
  for (auto [cfOperand, succVar] : llvm::zip_equal(cfOperands, succVars)) {
    (void)cfOperand;
    auto ba = dyn_cast<BlockArgument>(succVar);
    if (!ba || !isSpecialReg(ba.getType()) || !brOp)
      continue;
    impl->forwardingEdges.push_back(
        ForwardingEdge{brOp, ba, branchPoint.getSuccessorIndex()});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransformImpl
//===----------------------------------------------------------------------===//

void TransformImpl::handleBranchOpEdge(const ForwardingEdge &edge,
                                       RegisterTypeInterface sgprTy) {
  BranchOpInterface brOp = edge.brOp;
  SuccessorOperands succOperands =
      brOp.getSuccessorOperands(edge.successorIndex);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(brOp);

  // Insert a copy from the SREG to the SGPR, and update the successor operands.
  Value sgprSlot = createAllocation(rewriter, brOp.getLoc(), sgprTy);
  auto copyOp = lsir::CopyOp::create(rewriter, brOp.getLoc(), sgprSlot,
                                     succOperands[edge.destArg.getArgNumber()]);
  succOperands.getMutableForwardedOperands()[edge.destArg.getArgNumber()].set(
      copyOp.getTargetRes());
}

void TransformImpl::handleBlock(Block *bb, ArrayRef<Type> newArgTys,
                                ArrayRef<Location> bbArgLocs) {
  // Create the new block.
  Block *newBb = rewriter.createBlock(bb, newArgTys, bbArgLocs);
  SmallVector<Value> args;
  llvm::append_range(args, newBb->getArguments());

  // Insert the copies back from the SGPR to the actual register type.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newBb);
  for (auto &&[oldArg, arg] : llvm::zip_equal(bb->getArguments(), args)) {
    if (oldArg.getType() == arg.getType())
      continue;
    Value sreg =
        createAllocation(rewriter, oldArg.getLoc(),
                         cast<RegisterTypeInterface>(oldArg.getType()));
    lsir::CopyOp copyOp =
        lsir::CopyOp::create(rewriter, oldArg.getLoc(), sreg, arg);
    arg = copyOp.getTargetRes();
  }

  // Replace the old block with the new one.
  rewriter.replaceAllUsesWith(bb, newBb);
  rewriter.inlineBlockBefore(bb, newBb, newBb->end(), args);
}

LogicalResult TransformImpl::run() {
  // Collect the forwarding edges.
  Collector collector;
  collector.impl = this;
  if (failed(collector.walk(funcOp)))
    return failure();

  // If there are no forwarding edges, we're done.
  if (forwardingEdges.empty())
    return success();

  // Sort the edges to get a consistent lowering order within a compilation.
  llvm::stable_sort(forwardingEdges, [](ForwardingEdge a, ForwardingEdge b) {
    return std::make_tuple(a.destArg.getOwner(), a.destArg.getArgNumber(),
                           a.brOp.getOperation(), a.successorIndex) <
           std::make_tuple(b.destArg.getOwner(), b.destArg.getArgNumber(),
                           b.brOp.getOperation(), b.successorIndex);
  });

  // Update all edges.
  int64_t idx = 0, size = forwardingEdges.size();
  while (idx < size) {
    Block *tgtBb = forwardingEdges[idx].destArg.getOwner();

    // Find the last edge for this block.
    int64_t bbEnd = idx + 1;
    while (bbEnd < size && forwardingEdges[bbEnd].destArg.getOwner() == tgtBb)
      bbEnd++;

    // Get the block argument types.
    SmallVector<Type> bbArgTys;
    SmallVector<Location> bbArgLocs = llvm::map_to_vector(
        tgtBb->getArguments(), [](BlockArgument arg) { return arg.getLoc(); });
    llvm::append_range(bbArgTys, TypeRange(tgtBb->getArguments()));

    // Get the new block argument types.
    for (Type &argTy : bbArgTys) {
      if (!isSpecialReg(argTy))
        continue;
      int64_t sizeInBits = cast<RegisterTypeInterface>(argTy).getSizeInBits();
      assert(sizeInBits > 0 &&
             sizeInBits <= 32 * std::numeric_limits<int16_t>::max() &&
             "register size out of range");
      int16_t words = static_cast<int16_t>((sizeInBits + 31) / 32);
      argTy = getSGPR(argTy.getContext(), words);
    }

    // Rewrite all the branch ops.
    for (int64_t i = idx; i < bbEnd; i++) {
      ForwardingEdge &edge = forwardingEdges[i];
      handleBranchOpEdge(edge, cast<RegisterTypeInterface>(
                                   bbArgTys[edge.destArg.getArgNumber()]));
    }

    // Update the block.
    handleBlock(tgtBb, bbArgTys, bbArgLocs);

    // Move to the next block.
    idx = bbEnd;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LowerSRegBlockArgs
//===----------------------------------------------------------------------===//

void LowerSRegBlockArgs::runOnOperation() {
  getOperation()->walk<WalkOrder::PostOrder>([&](FunctionOpInterface funcOp) {
    if (failed(TransformImpl(funcOp).run())) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
}
