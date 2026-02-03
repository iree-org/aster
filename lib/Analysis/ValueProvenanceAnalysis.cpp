//===- ValueProvenanceAnalysis.cpp - Value provenance analysis ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "value-provenance-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// ValueProvenance
//===----------------------------------------------------------------------===//

ValueProvenance ValueProvenance::join(const ValueProvenance &lhs,
                                      const ValueProvenance &rhs) {
  if (lhs.isTop() || rhs.isTop())
    return getTop();

  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;

  if (*lhs.provenanceAllocas == *rhs.provenanceAllocas)
    return lhs;

  // Phi-coalescing: merge allocas that must share a register.
  llvm::SetVector<Value> merged;
  merged.insert(lhs.provenanceAllocas->begin(), lhs.provenanceAllocas->end());
  merged.insert(rhs.provenanceAllocas->begin(), rhs.provenanceAllocas->end());

  ValueProvenance result;
  result.provenanceAllocas = std::move(merged);
  return result;
}

void ValueProvenance::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<TOP>";
    return;
  }
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  os << "[";
  llvm::interleaveComma(*provenanceAllocas, os, [&](Value v) { os << v; });
  os << "]";
}

//===----------------------------------------------------------------------===//
// ValueProvenanceAnalysis
//===----------------------------------------------------------------------===//

// TODO: make this a normal form.
LogicalResult ValueProvenanceAnalysis::verifyIRForm(Operation *topOp) {
  LogicalResult result = success();

  topOp->walk([&](Operation *op) {
    // lsir.from_reg/to_reg break provenance tracking.
    if (isa<lsir::FromRegOp, lsir::ToRegOp>(op)) {
      op->emitError() << "unsupported op for value provenance analysis; "
                      << "from_reg/to_reg must be eliminated first";
      result = failure();
    }
  });

  return result;
}

LogicalResult ValueProvenanceAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::Lattice<ValueProvenance> *> operandLattices,
    ArrayRef<dataflow::Lattice<ValueProvenance> *> results) {

  LDBG() << "Visiting: " << *op;

  // TODO: Some interface should be defined for this.
  if (isa<AllocaOp, lsir::AllocaOp>(op)) {
    Value result = op->getResult(0);
    propagateIfChanged(results[0], results[0]->join(ValueProvenance(result)));
    return success();
  }

  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (OpOperand &outsOperand : instOp.getInstOutsMutable()) {
      size_t idx = outsOperand.getOperandNumber();
      propagateIfChanged(results[idx],
                         results[idx]->join(operandLattices[idx]->getValue()));
    }
    return success();
  }

  if (auto castOp = dyn_cast<DeallocCastOp>(op)) {
    propagateIfChanged(results[0],
                       results[0]->join(operandLattices[0]->getValue()));
    return success();
  }

  // Other ops don't originate from allocas.
  setAllToEntryStates(results);

  return success();
}

void ValueProvenanceAnalysis::setToEntryState(
    dataflow::Lattice<ValueProvenance> *lattice) {
  propagateIfChanged(lattice, lattice->join(ValueProvenance::getTop()));
}

ValueProvenanceAnalysis *ValueProvenanceAnalysis::create(DataFlowSolver &solver,
                                                         Operation *topOp) {
  dataflow::loadBaselineAnalyses(solver);
  auto *analysis = solver.load<ValueProvenanceAnalysis>();
  if (failed(solver.initializeAndRun(topOp)))
    return nullptr;

  // Build index mappings for all allocas (deterministic order from IR walk).
  int64_t allocaIdx = 0;
  topOp->walk([&](Operation *op) {
    if (isa<AllocaOp, lsir::AllocaOp>(op)) {
      Value result = op->getResult(0);
      analysis->allocaToIndex[result] = allocaIdx;
      analysis->indexToAlloca.push_back(result);
      analysis->phiEquivalences.insert(allocaIdx);
      allocaIdx++;
    }
  });

  // Extract phiEquivalences from block args where multiple allocas merged.
  topOp->walk([&](Block *block) {
    for (BlockArgument arg : block->getArguments()) {
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ValueProvenance>>(arg);
      if (!lattice)
        continue;

      ArrayRef<Value> allocas = lattice->getValue().getAllocas();
      if (allocas.size() <= 1)
        continue;

      int64_t firstIdx = analysis->allocaToIndex.lookup(allocas[0]);
      for (Value a : allocas.drop_front()) {
        int64_t idx = analysis->allocaToIndex.lookup(a);
        analysis->phiEquivalences.unionSets(firstIdx, idx);
        LDBG() << "Unioning allocas: " << a << " with " << allocas[0];
      }
    }
  });

  return analysis;
}

FailureOr<Value>
ValueProvenanceAnalysis::getCanonicalPhiEquivalentAlloca(Value v) const {
  auto *lattice = solver.lookupState<dataflow::Lattice<ValueProvenance>>(v);
  assert(lattice && "expected that value provenance was computed");

  const ValueProvenance &prov = lattice->getValue();
  if (prov.isTop() || prov.isUninitialized())
    return failure();

  Value alloca = prov.getFirstAllocaInSet();
  if (!alloca)
    return failure();

  // TODO: could avoid extra looking by storing {Value, int64_t} in the lattice.
  // This is a performance / memory trade-off.
  auto it = allocaToIndex.find(alloca);
  assert(it != allocaToIndex.end() && "expected alloca to be in index map");

  int64_t leaderIdx = phiEquivalences.getLeaderValue(it->second);
  return indexToAlloca[leaderIdx];
}

bool ValueProvenanceAnalysis::arePhiEquivalent(Value a, Value b) const {
  auto itA = allocaToIndex.find(a);
  auto itB = allocaToIndex.find(b);
  if (itA == allocaToIndex.end() || itB == allocaToIndex.end())
    return false;
  return phiEquivalences.isEquivalent(itA->second, itB->second);
}

SmallVector<Value>
ValueProvenanceAnalysis::getPhiEquivalentAllocas(Value alloca) const {
  auto it = allocaToIndex.find(alloca);
  if (it == allocaToIndex.end())
    return {};

  SmallVector<Value> result;
  for (int64_t idx : phiEquivalences.members(it->second))
    result.push_back(indexToAlloca[idx]);
  return result;
}
