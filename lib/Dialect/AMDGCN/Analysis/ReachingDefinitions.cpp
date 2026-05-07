//===- ReachingDefinitions.cpp - Reaching definitions analysis ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/InstImpl.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// ReachingDefinitionsState
//===----------------------------------------------------------------------===//

ChangeResult
ReachingDefinitionsState::join(const ReachingDefinitionsState &other) {
  ChangeResult changed = ChangeResult::NoChange;
  for (const auto &[allocation, otherBucket] : other.definitions) {
    auto [it, inserted] = definitions.insert({allocation, {}});
    BucketTy &bucket = it->second;
    // New entry, copy the other bucket.
    if (inserted) {
      bucket = otherBucket;
      if (!bucket.empty())
        changed = ChangeResult::Change;
      continue;
    }
    // Existing entry: merge. Linear-scan wins for small bucket sizes; above 8
    // use a SmallPtrSet to avoid pathological O(N^2).
    constexpr size_t kLinearScanThreshold = 8;
    if (bucket.size() <= kLinearScanThreshold) {
      for (OpOperand *opnd : otherBucket) {
        if (llvm::is_contained(bucket, opnd))
          continue;
        bucket.push_back(opnd);
        changed = ChangeResult::Change;
      }
    } else {
      SmallPtrSet<OpOperand *, 16> seen(bucket.begin(), bucket.end());
      for (OpOperand *opnd : otherBucket) {
        if (!seen.insert(opnd).second)
          continue;
        bucket.push_back(opnd);
        changed = ChangeResult::Change;
      }
    }
  }
  return changed;
}

ChangeResult ReachingDefinitionsState::killDefinitions(Value allocation) {
  return definitions.erase(allocation) ? ChangeResult::Change
                                       : ChangeResult::NoChange;
}

ChangeResult ReachingDefinitionsState::addDefinition(Value allocation,
                                                     OpOperand *definition) {
  assert(definition && "null OpOperand in addDefinition");
  BucketTy &bucket = definitions[allocation];
  if (llvm::is_contained(bucket, definition))
    return ChangeResult::NoChange;
  bucket.push_back(definition);
  return ChangeResult::Change;
}

static void printEntry(raw_ostream &os, Value allocation, OpOperand *opnd) {
  os << "{" << ValueWithFlags(allocation, true) << ", "
     << OpWithFlags(opnd->getOwner(), OpPrintingFlags().skipRegions()) << "<"
     << opnd->getOperandNumber() << ">}";
}

void ReachingDefinitionsState::print(raw_ostream &os) const {
  if (definitions.empty()) {
    os << "[]";
    return;
  }
  // DenseMap iteration order is non-deterministic; sort by opaque pointer for
  // stable output within a single run. This is still non-deterministic across
  // platforms. For full determinism use:
  //   ReachingDefinitionsState::print(os, ssaMap, dominance).
  SmallVector<Value> keys;
  keys.reserve(definitions.size());
  for (const auto &[k, _] : definitions)
    keys.push_back(k);
  llvm::sort(keys, [](Value a, Value b) {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  });
  os << "[";
  bool firstEntry = true;
  for (Value allocation : keys) {
    for (OpOperand *opnd : definitions.find(allocation)->second) {
      if (!firstEntry)
        os << ", ";
      firstEntry = false;
      printEntry(os, allocation, opnd);
    }
  }
  os << "]";
}

void ReachingDefinitionsState::print(raw_ostream &os,
                                     const mlir::aster::SSAMap &ssaMap,
                                     const DominanceInfo &dominance) const {
  if (definitions.empty()) {
    os << "[]";
    return;
  }
  using Entry = std::pair<Value, OpOperand *>;
  SmallVector<Entry, 8> sorted;
  for (const auto &[allocation, bucket] : definitions)
    for (OpOperand *opnd : bucket)
      sorted.emplace_back(allocation, opnd);
  llvm::sort(sorted, [&](const Entry &a, const Entry &b) {
    int64_t idA = ssaMap.lookup(a.first);
    int64_t idB = ssaMap.lookup(b.first);
    if (idA != idB)
      return idA < idB;
    Operation *opA = a.second->getOwner();
    Operation *opB = b.second->getOwner();
    if (opA != opB)
      return dominance.dominates(opA, opB);
    return a.second->getOperandNumber() < b.second->getOperandNumber();
  });
  os << "[\n";
  llvm::interleave(
      sorted, os,
      [&](const Entry &entry) {
        os << "  {" << ssaMap.lookup(entry.first) << " = `"
           << ValueWithFlags(entry.first, true) << "`, "
           << OpWithFlags(entry.second->getOwner(),
                          OpPrintingFlags().skipRegions())
           << "<" << entry.second->getOperandNumber() << ">}";
      },
      "\n");
  os << "\n]";
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::ReachingDefinitionsState)

//===----------------------------------------------------------------------===//
// ReachingDefinitionsAnalysis
//===----------------------------------------------------------------------===//

void ReachingDefinitionsAnalysis::setToEntryState(
    ReachingDefinitionsState *lattice) {
  propagateIfChanged(lattice, lattice->setToEntryState());
}

LogicalResult ReachingDefinitionsAnalysis::visitOperation(
    Operation *op, const ReachingDefinitionsState &before,
    ReachingDefinitionsState *after) {
  // Start with the state before this operation.
  ChangeResult changed = after->join(before);

  auto _ = llvm::make_scope_exit([&]() { propagateIfChanged(after, changed); });

  // Only consider InstOpInterface effects.
  auto instOp = dyn_cast<InstOpInterface>(op);
  if (!instOp)
    return success();

  // If provided allow the callback to kill definitions.
  if (killCallback) {
    auto killDefs = [&](ValueRange values) {
      for (Value value : values)
        changed |= after->killDefinitions(value);
    };

    if (failed(killCallback(instOp, killDefs)))
      return failure();
  }

  OperandRange operands = instOp.getInstOuts();
  if (operands.empty())
    return success();

  int64_t startOperand = operands.getBeginOperandIndex();
  bool filterOut = definitionFilter && !definitionFilter(op);
  for (OpOperand &operand :
       op->getOpOperands().slice(startOperand, operands.size())) {

    assert((!isa<RegisterTypeInterface>(operand.get().getType()) ||
            !cast<RegisterTypeInterface>(operand.get().getType())
                 .hasValueSemantics()) &&
           "IR is not in post-to-register-semantics DPS normal form");

    // Get the allocas behind the operand.
    FailureOr<ValueRange> allocas = getAllocasOrFailure(operand.get());
    if (failed(allocas))
      return failure();

    // Kill previous definitions to this allocation, then add this definition.
    for (Value alloc : *allocas) {
      /// Note: we always have to kill definitions, even if we filter out.
      changed |= after->killDefinitions(alloc);
      if (filterOut)
        continue;
      changed |= after->addDefinition(alloc, &operand);
    }
  }
  return success();
}

/// Verify that all InstOpInterface `outs` operands in `root` are storage-
/// semantic (not value-semantic). This is the post-to-register-semantics DPS
/// normal form precondition.
static LogicalResult
verifyPostToRegisterSemanticsDPSNormalForm(Operation *root) {
  WalkResult result = root->walk([](InstOpInterface instOp) -> WalkResult {
    for (Value operand : instOp.getInstOuts()) {
      auto regTy = dyn_cast<RegisterTypeInterface>(operand.getType());
      if (regTy && regTy.hasValueSemantics())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

FailureOr<ReachingDefinitionsAnalysis *> ReachingDefinitionsAnalysis::create(
    DataFlowSolver &solver, Operation *root,
    llvm::function_ref<bool(Operation *)> definitionFilter,
    llvm::function_ref<LogicalResult(InstOpInterface, KillDefsFn)>
        killCallback) {
  if (failed(verifyPostToRegisterSemanticsDPSNormalForm(root)))
    return failure();
  return solver.load<ReachingDefinitionsAnalysis>(definitionFilter,
                                                  killCallback);
}

//===----------------------------------------------------------------------===//
// hasReachingLoadDefinition via on-demand backwards walk
//===----------------------------------------------------------------------===//

namespace {
/// Per-mov backwards reachability search, computed lazily without a fixpoint.
/// Operates on allocations post-bufferization (not in SSA form) and requires
/// `#amdgcn.no_op_with_regions` (flat-CFG: every block reachable via
/// `Block::getPredecessors()`).
struct LoadReachability {
  Value allocation;
  Operation *queryOp;
  DenseSet<Block *> seenBlocks;
  DenseMap<Block *, Operation *> latestWriterPerBlock;

  LoadReachability(Value allocation, Operation *queryOp)
      : allocation(allocation), queryOp(queryOp) {
    collectLatestWriterPerBlock();
  }

  void collectLatestWriterPerBlock() {
    for (OpOperand &use : allocation.getUses()) {
      if (isa<OperandBundleOpInterface>(use.getOwner())) {
        // One-hop bundle indirection: ops can write `allocation` through a
        // bundle wrapper (e.g. `amdgcn.make_register_range`).
        for (Value bundleResult : use.getOwner()->getResults())
          for (OpOperand &bundleUse : bundleResult.getUses())
            recordIfLatestWriterInBlock(bundleUse);
        continue;
      }
      recordIfLatestWriterInBlock(use);
    }
  }

  void recordIfLatestWriterInBlock(OpOperand &use) {
    Operation *user = use.getOwner();
    auto inst = dyn_cast<InstOpInterface>(user);
    if (!inst)
      return;
    OperandRange outs = inst.getInstOuts();
    if (outs.empty())
      return;
    unsigned outsBegin = outs.getBeginOperandIndex();
    unsigned idx = use.getOperandNumber();
    // Write => operand number is in the `outs` range.
    if (idx < outsBegin || idx >= outsBegin + outs.size())
      return;
    if (user->getBlock() == queryOp->getBlock() &&
        !user->isBeforeInBlock(queryOp))
      return;
    Operation *&latest = latestWriterPerBlock[user->getBlock()];
    if (!latest || latest->isBeforeInBlock(user))
      latest = user;
  }

  /// O(1) lookup: each path's relevant writer is precomputed.
  ///   load     -> true  (path contributes "yes")
  ///   non-load -> false (path is killed: do not continue further back)
  ///   none     -> nullopt (recurse to predecessors)
  std::optional<bool> latestWriterIsLoad(Block *block) {
    auto it = latestWriterPerBlock.find(block);
    if (it == latestWriterPerBlock.end())
      return std::nullopt;
    return isa<LoadOpInterface>(it->second);
  }

  bool reachesBefore(Block *block) {
    if (auto r = latestWriterIsLoad(block))
      return *r;
    return reachesEntryOf(block);
  }

  /// Predecessor walk under the flat-CFG precondition
  /// (`#amdgcn.no_op_with_regions`). All control flow lives at the block level
  /// via `cf.br`/`cf.cond_br`, so `Block::getPredecessors()` is the COMPLETE
  /// predecessor set; the structured-op recursion required by the general MLIR
  /// form is dead here. See `key_insights/control-flow-traversal-forms.md`.
  bool reachesEntryOf(Block *block) {
    if (!seenBlocks.insert(block).second)
      return false;
    for (Block *pred : block->getPredecessors())
      if (reachesBefore(pred))
        return true;
    return false;
  }
};
} // namespace

/// Returns true if some path from the function entry to `mov` has its
/// most-recent writer of `allocation` come from a `LoadOpInterface` op.
bool mlir::aster::amdgcn::hasReachingLoadDefinition(Operation *mov,
                                                    Value allocation) {
  Block *block = mov->getBlock();
  if (!block)
    return false;
  LoadReachability query(allocation, mov);
  return query.reachesBefore(block);
}
