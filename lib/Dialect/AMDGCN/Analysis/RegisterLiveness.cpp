//===- RegisterLiveness.cpp - Register liveness analysis ------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RegisterLiveness.h"
#include "aster/Dialect/AMDGCN/Analysis/Utils.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "amdgcn-register-liveness"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// RegisterLivenessState
//===----------------------------------------------------------------------===//

void RegisterLivenessState::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<top>";
    return;
  }
  if (isEmpty()) {
    os << "[]";
    return;
  }
  const ValueSet *values = getLiveValues();

  os << "[";
  llvm::interleaveComma(*values, os, [&](Value value) {
    value.printAsOperand(os, OpPrintingFlags());
  });
  os << "]";
}

void RegisterLivenessState::print(raw_ostream &os, const SSAMap &ssaMap) const {
  if (isTop()) {
    os << "<top>";
    return;
  }
  if (isEmpty()) {
    os << "[]";
    return;
  }
  const ValueSet *values = getLiveValues();
  // Sort the ids to make the output deterministic.
  SmallVector<std::pair<Value, int64_t>> ids;
  ssaMap.getIds(*values, ids);
  llvm::sort(ids, [](const std::pair<Value, int64_t> &lhs,
                     const std::pair<Value, int64_t> &rhs) {
    return lhs.second < rhs.second;
  });
  os << "[";
  llvm::interleaveComma(ids, os, [&](const std::pair<Value, int64_t> &entry) {
    os << entry.second << " = `" << ValueWithFlags(entry.first, true) << "`";
  });
  os << "]";
}

ChangeResult RegisterLivenessState::meet(const RegisterLivenessState &lattice) {
  // Empty lattice contributes nothing.
  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  // Top absorbs everything.
  if (isTop())
    return ChangeResult::NoChange;

  // Meet with top results in top.
  if (lattice.isTop())
    return setToTop();

  // Both have concrete values - compute union.
  const ValueSet *otherValues = lattice.getLiveValues();

  // Initialize if we have no values yet.
  ChangeResult changed = ChangeResult::NoChange;

  // Compute union.
  size_t oldSize = liveValues->size();
  liveValues->insert(otherValues->begin(), otherValues->end());
  if (liveValues->size() != oldSize)
    changed = ChangeResult::Change;

  return changed;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::RegisterLivenessState)

//===----------------------------------------------------------------------===//
// RegisterLiveness
//===----------------------------------------------------------------------===//

#define DUMP_STATE_HELPER(name, obj)                                           \
  LDBG_OS([&](raw_ostream &os) {                                               \
    os << "Visiting " name ": " << obj << "\n";                                \
    os << "  Incoming lattice: ";                                              \
    after.print(os);                                                           \
  });                                                                          \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "  Outgoing lattice: ";                                            \
      before->print(os);                                                       \
    });                                                                        \
  });

void RegisterLiveness::transferFunction(const RegisterLivenessState &after,
                                        RegisterLivenessState *before,
                                        ValueRange deadValues,
                                        ValueRange inValues) {
  RegisterLivenessState::ValueSet *liveValues = before->getLiveValues();
  if (!liveValues)
    return;

  // Meet with the after state.
  ChangeResult changed = before->meet(after);

  // Remove the dead values.
  for (Value deadValue : deadValues) {
    changed |= liveValues->erase(deadValue) ? ChangeResult::Change
                                            : ChangeResult::NoChange;
  }

  // Add the in values.
  int64_t size = liveValues->size();
  liveValues->insert_range(inValues);
  if (liveValues->size() != size)
    changed = ChangeResult::Change;
  propagateIfChanged(before, changed);
}

bool RegisterLiveness::handleTopPropagation(const RegisterLivenessState &after,
                                            RegisterLivenessState *before) {
  if (after.isTop() || before->isTop()) {
    propagateIfChanged(before, before->setToTop());
    return true;
  }
  return false;
}

/// Check if any of the values in the range have value semantics.
static bool checkValueSemantics(ValueRange values) {
  if (values.empty())
    return false;
  return llvm::any_of(values, [](Value v) {
    auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
    return regTy && regTy.hasValueSemantics();
  });
}

LogicalResult
RegisterLiveness::visitOperation(Operation *op,
                                 const RegisterLivenessState &after,
                                 RegisterLivenessState *before) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));

  // Check if the operation has value semantics.
  if (checkValueSemantics(op->getResults()) ||
      checkValueSemantics(op->getOperands())) {
    incompleteLiveness = true;
  }

  // Handle top propagation.
  if (handleTopPropagation(after, before))
    return success();

  // Handle instruction operations.
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    SmallVector<Value, 4> outs;
    if (failed(getAllocasOrFailure(instOp.getInstOuts(), outs))) {
      LDBG() << "Failed to get allocas for values "
             << llvm::interleaved_array(instOp.getInstOuts());
      return failure();
    }
    SmallVector<Value, 4> ins;
    if (failed(getAllocasOrFailure(instOp.getInstIns(), ins))) {
      LDBG() << "Failed to get allocas for values "
             << llvm::interleaved_array(instOp.getInstIns());
      return failure();
    }
    transferFunction(after, before, outs, ins);
    return success();
  }

  // Handle alloca operations.
  if (auto aOp = dyn_cast<AllocaOp>(op)) {
    transferFunction(after, before, aOp.getResult(), {});
    return success();
  }

  // Pass through the after state.
  propagateIfChanged(before, before->meet(after));
  return success();
}

void RegisterLiveness::visitBlockTransfer(Block *block, ProgramPoint *point,
                                          Block *successor,
                                          const RegisterLivenessState &after,
                                          RegisterLivenessState *before) {
  DUMP_STATE_HELPER("block", block);
  if (handleTopPropagation(after, before))
    return;

  // Check if any of the block arguments have value semantics.
  if (checkValueSemantics(successor->getArguments()) ||
      checkValueSemantics(block->getArguments()))
    incompleteLiveness = true;

  transferFunction(after, before,
                   llvm::to_vector_of<Value>(successor->getArguments()), {});
}

void RegisterLiveness::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const RegisterLivenessState &after, RegisterLivenessState *before) {
  assert(action == dataflow::CallControlFlowAction::ExternalCallee &&
         "we don't support inter-procedural analysis");
  bool hasRegisterOperands = llvm::any_of(call.getArgOperands(), [](Value v) {
    return isa<RegisterTypeInterface>(v.getType());
  });
  bool hasRegisterResults = llvm::any_of(call->getResultTypes(), [](Type t) {
    return isa<RegisterTypeInterface>(t);
  });
  if (hasRegisterOperands || hasRegisterResults)
    propagateIfChanged(before, before->setToTop());
  propagateIfChanged(before, before->meet(after));
}

void RegisterLiveness::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionSuccessor regionTo, const RegisterLivenessState &after,
    RegisterLivenessState *before) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));
  if (handleTopPropagation(after, before))
    return;

  ValueRange inputs = branch.getSuccessorInputs(regionTo);
  ValueRange operands = branch.getSuccessorOperands(regionFrom, regionTo);

  // Check if any of the block arguments have value semantics.
  if (checkValueSemantics(inputs) || checkValueSemantics(operands))
    incompleteLiveness = true;

  transferFunction(after, before, inputs, {});
}

void RegisterLiveness::setToExitState(RegisterLivenessState *lattice) {
  // At exit points, nothing is live initially.
  propagateIfChanged(lattice, lattice->setToEmpty());
}
