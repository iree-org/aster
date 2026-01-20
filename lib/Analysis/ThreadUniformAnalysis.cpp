//===- ThreadUniformAnalysis.cpp - Thread uniform analysis ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "aster-thread-uniform-analysis"

namespace mlir::aster::dataflow {
//===----------------------------------------------------------------------===//
// ThreadUniform
//===----------------------------------------------------------------------===//

ThreadUniform ThreadUniform::join(const ThreadUniform &lhs,
                                  const ThreadUniform &rhs) {
  if (lhs.state == dependent || rhs.state == dependent)
    return getDependent();
  return (lhs.state == uniform || rhs.state == uniform) ? getUniform()
                                                        : ThreadUniform();
}

void ThreadUniform::print(llvm::raw_ostream &s) const {
  s << (state == uninitialized ? "uninitialized"
                               : (state == uniform ? "uniform" : "dependent"));
}

//===----------------------------------------------------------------------===//
// ThreadUniformAnalysis
//===----------------------------------------------------------------------===//
/// This function is an overly conservative estimate ops that are safe to assume
/// to be uniform. TODO: Encode this in an interface.
static bool isWorkgroupUniform(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (isa<gpu::BlockDimOp, gpu::BlockIdOp, aster_utils::GridDimOp,
          aster_utils::BlockIdOp, aster_utils::BlockDimOp,
          lsir::AssumeNoaliasOp, aster_utils::AssumeRangeOp,
          aster_utils::AssumeUniformOp>(op))
    return true;
  if (isa<affine::AffineDialect>(op->getDialect()))
    return !isa<affine::AffineDmaStartOp, affine::AffineDmaWaitOp>(op);
  return isa<arith::ArithDialect>(op->getDialect()) &&
         op->getNumResults() == 1 &&
         op->getResult(0).getType().isIntOrIndexOrFloat();
}

LogicalResult ThreadUniformAnalysis::visitOperation(
    Operation *op, ArrayRef<const ThreadUniformLattice *> operands,
    ArrayRef<ThreadUniformLattice *> results) {
  // Early exit if any of the operands is already dependent.
  if (llvm::any_of(operands, [&](const ThreadUniformLattice *lattice) {
        return ThreadUniform::getDependent() == lattice->getValue();
      })) {
    for (ThreadUniformLattice *v : results) {
      propagateIfChanged(v, v->join(ThreadUniform::getDependent()));
    }
    LDBG() << " dependent op: " << *op;
    return success();
  }

  // Check if it's a uniform op.
  if (isWorkgroupUniform(op)) {
    LDBG() << " uniform op: " << *op;
    for (ThreadUniformLattice *v : results) {
      propagateIfChanged(v, v->join(ThreadUniform::getUniform()));
    }
    return success();
  }

  LDBG() << " pessimistic dependent op: " << *op;
  // Be pessimistic about all other ops.
  setAllToEntryStates(results);
  return success();
}

void ThreadUniformAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor, ValueRange successorInputs,
    ArrayRef<ThreadUniformLattice *> argLattices, unsigned firstIndex) {
  auto loop = dyn_cast<LoopLikeOpInterface>(op);

  // Be pessimistic about non loop ops.
  if (!loop) {
    SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, successorInputs, argLattices, firstIndex);
    return;
  }

  // Get the induction variables, and be pessimistic if they cannot be
  // retrieved.
  std::optional<SmallVector<Value>> iV = loop.getLoopInductionVars();
  if (!iV) {
    SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, successorInputs, argLattices, firstIndex);
    return;
  }

  // Get the inits.
  OperandRange inits = loop.getInits();
  assert((iV->size() + inits.size() == argLattices.size() ||
          inits.size() == argLattices.size()) &&
         "unsupported loop-like op");

  // Get the loop structure.
  std::optional<SmallVector<OpFoldResult>> lb = loop.getLoopLowerBounds();
  std::optional<SmallVector<OpFoldResult>> ub = loop.getLoopUpperBounds();
  std::optional<SmallVector<OpFoldResult>> sv = loop.getLoopSteps();
  assert(lb && ub && sv && "unsupported loop-like op");
  assert((iV->front() == argLattices.front()->getAnchor() ||
          (inits.empty() ||
           op->getResult(0) == argLattices.front()->getAnchor())) &&
         "unsupported loop-like op");

  // Helper function to get the state of an op fold result.
  auto getState = [&](OpFoldResult ofr) {
    auto v = dyn_cast<Value>(ofr);
    if (!v)
      return ThreadUniform::getUniform();
    return getLatticeElement(v)->getValue();
  };

  // Get whether the loop is thread uniform based on the structure.
  ThreadUniform value;
  for (auto [lv, uv, s] : llvm::zip(*lb, *ub, *sv)) {
    value = ThreadUniform::join(value, getState(lv));
    value = ThreadUniform::join(value, getState(uv));
    value = ThreadUniform::join(value, getState(s));
  }

  // This is needed because dataflow is broken, and will call this function to
  // check the results of a control-flow op. TODO: fix dataflow upstream.
  if (inits.size() == argLattices.size()) {
    for (auto [lattice, operand] :
         llvm::zip(argLattices, loop.getRegionIterArgs())) {
      join(lattice, *getLatticeElement(operand));
      propagateIfChanged(lattice, lattice->join(value));
    }
    return;
  }

  // Propagate the state.
  for (ThreadUniformLattice *lattice : argLattices.take_front(iV->size())) {
    propagateIfChanged(lattice, lattice->join(value));
  }
  for (auto [lattice, operand] :
       llvm::zip(argLattices.drop_front(iV->size()), inits)) {
    join(lattice, *getLatticeElement(operand));
    propagateIfChanged(lattice, lattice->join(value));
  }
}

void ThreadUniformAnalysis::visitCallableOperation(
    CallableOpInterface callable,
    ArrayRef<mlir::dataflow::AbstractSparseLattice *> argLattices) {
  // Mark kernel args as uniform.
  if (auto gpuFunc = dyn_cast<aster::GPUFuncInterface>(callable.getOperation());
      gpuFunc && gpuFunc.isGPUKernel()) {
    for (mlir::dataflow::AbstractSparseLattice *lattice : argLattices) {
      propagateIfChanged(
          lattice, reinterpret_cast<ThreadUniformLattice *>(lattice)->join(
                       ThreadUniform::getUniform()));
    }
    return;
  }

  return AbstractSparseForwardDataFlowAnalysis::visitCallableOperation(
      callable, argLattices);
}
} // namespace mlir::aster::dataflow

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::dataflow::ThreadUniformLattice)
