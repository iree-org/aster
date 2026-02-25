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
#include "aster/IR/InstImpl.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
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
  for (const Definition &def : other.definitions) {
    if (definitions.insert(def).second)
      changed = ChangeResult::Change;
  }
  return changed;
}

ChangeResult ReachingDefinitionsState::killDefinitions(Value allocation) {
  ChangeResult changed = ChangeResult::NoChange;
  auto lb = definitions.lower_bound(Definition::createLowerBound(allocation));
  while (lb != definitions.end() && lb->allocation == allocation) {
    lb = definitions.erase(lb);
    changed = ChangeResult::Change;
  }
  return changed;
}

ChangeResult ReachingDefinitionsState::addDefinition(Definition definition) {
  assert(definition.definition && "Definition must have an operand");
  return definitions.insert(definition).second ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

void ReachingDefinitionsState::print(raw_ostream &os) const {
  os << "[";
  llvm::interleaveComma(definitions, os, [&](const Definition &def) {
    os << "{" << ValueWithFlags(def.allocation, true) << ", ";
    assert(def.definition && "Definition must have an operand");
    os << OpWithFlags(def.definition->getOwner(),
                      OpPrintingFlags().skipRegions())
       << "<" << def.definition->getOperandNumber() << ">";
  });
  os << "]";
}

void ReachingDefinitionsState::print(raw_ostream &os,
                                     const mlir::aster::SSAMap &ssaMap,
                                     const DominanceInfo &dominance) const {
  if (definitions.empty()) {
    os << "[]";
    return;
  }
  SmallVector<Definition, 8> sorted(definitions.begin(), definitions.end());
  llvm::sort(sorted, [&](const Definition &a, const Definition &b) {
    int64_t idA = ssaMap.lookup(a.allocation);
    int64_t idB = ssaMap.lookup(b.allocation);
    if (idA != idB)
      return idA < idB;
    assert(a.definition && "Definition must have an operand");
    assert(b.definition && "Definition must have an operand");
    Operation *opA = a.definition->getOwner();
    Operation *opB = b.definition->getOwner();
    if (opA != opB)
      return dominance.dominates(opA, opB);
    return a.definition->getOperandNumber() < b.definition->getOperandNumber();
  });
  os << "[\n";
  llvm::interleave(
      sorted, os,
      [&](const Definition &def) {
        os << "  {" << ssaMap.lookup(def.allocation) << " = `"
           << ValueWithFlags(def.allocation, true) << "`, ";
        assert(def.definition && "Definition must have an operand");
        os << OpWithFlags(def.definition->getOwner(),
                          OpPrintingFlags().skipRegions())
           << "<" << def.definition->getOperandNumber() << ">}";
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
      changed |= after->addDefinition(Definition{alloc, operand});
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
