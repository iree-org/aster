//===- DPSAliasAnalysis.cpp - DPS alias analysis --------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Analysis/ValueProvenanceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>

#define DEBUG_TYPE "dps-alias-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// AliasEquivalenceClass
//===----------------------------------------------------------------------===//

void AliasEquivalenceClass::print(raw_ostream &os) const {
  switch (state) {
  case State::Uninitialized:
    os << "<UNINITIALIZED>";
    return;
  case State::Known:
    os << "[";
    llvm::interleaveComma(eqClassIds, os);
    os << "]";
    return;
  case State::Unknown:
    os << "<UNKNOWN>";
    return;
  case State::Conflict:
    os << "<CONFLICT>";
    return;
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AliasEquivalenceClass &eqClass) {
  eqClass.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// DPSAliasAnalysis
//===----------------------------------------------------------------------===//

/// Check operands for CONFLICT state (will trip register allocation) and
/// UNKNOWN state (must be handled conservatively).
/// Returns true if any register-like operand has CONFLICT state.
static bool checkOperandStates(
    SmallVectorImpl<Value> &conflictingValues,
    SmallVectorImpl<Value> &unknownValues,
    ArrayRef<const dataflow::Lattice<AliasEquivalenceClass> *> lattices,
    ValueRange operands) {
  bool hasConflict = false;
  for (auto [operand, lattice] : llvm::zip_equal(operands, lattices)) {
    if (!isa<RegisterTypeInterface>(operand.getType()))
      continue;
    const auto &val = lattice->getValue();
    if (val.isConflict()) {
      conflictingValues.push_back(operand);
      hasConflict = true;
    } else if (val.isUnknown()) {
      unknownValues.push_back(operand);
    }
  }
  return hasConflict;
}

/// Check results for CONFLICT state (will trip register allocation) and
/// UNKNOWN state (must be handled conservatively).
/// Returns true if any register-like result has CONFLICT state.
static bool
checkResultStates(SmallVectorImpl<Value> &conflictingValues,
                  SmallVectorImpl<Value> &unknownValues,
                  ArrayRef<dataflow::Lattice<AliasEquivalenceClass> *> lattices,
                  ValueRange results) {
  bool hasConflict = false;
  for (auto [result, lattice] : llvm::zip_equal(results, lattices)) {
    if (!isa<RegisterTypeInterface>(result.getType()))
      continue;
    const auto &val = lattice->getValue();
    if (val.isConflict()) {
      conflictingValues.push_back(result);
      hasConflict = true;
    } else if (val.isUnknown()) {
      unknownValues.push_back(result);
    }
  }
  return hasConflict;
}

LogicalResult DPSAliasAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::Lattice<AliasEquivalenceClass> *> operandLattices,
    ArrayRef<dataflow::Lattice<AliasEquivalenceClass> *> results) {
  // Track result states at exit for diagnostics.
  auto _atExit = llvm::make_scope_exit([&]() {
    if (ValueRange vals = op->getResults(); !vals.empty())
      checkResultStates(conflictingValues, unknownValues, results, vals);

    // Log the lattices at exit for debugging.
    LDBG_OS([&](llvm::raw_ostream &os) {
      os << "Alias sets for op: " << *op << " =>\n  ";
      llvm::interleaveComma(llvm::enumerate(results), os, [&](auto idxLattice) {
        os << idxLattice.index() << ": ";
        idxLattice.value()->getValue().print(os);
      });
    });
  });

  // Check operand states. If any operand has CONFLICT, propagate it.
  bool hasConflict = checkOperandStates(conflictingValues, unknownValues,
                                        operandLattices, op->getOperands());
  if (hasConflict) {
    for (dataflow::Lattice<AliasEquivalenceClass> *result : results)
      propagateIfChanged(result,
                         result->join(AliasEquivalenceClass::getConflict()));
    return success();
  }

  // AllocaOps define new equivalence classes (may be phi-coalesced).
  if (isa<AllocaOp, lsir::AllocaOp>(op)) {
    EqClassID eqClassId = getOrCreateEqClassId(op->getResult(0));
    propagateIfChanged(results[0],
                       results[0]->join(AliasEquivalenceClass({eqClassId})));
    return success();
  }

  // InstOpInterface: results inherit alias classes from DPS outs operands.
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      size_t idx = operand.getOperandNumber();
      propagateIfChanged(results[idx],
                         results[idx]->join(operandLattices[idx]->getValue()));
    }
    return success();
  }

  // MakeRegisterRangeOp: result contains all operand alias classes.
  if (isa<amdgcn::MakeRegisterRangeOp>(op)) {
    AliasEquivalenceClass::EqClassList eqClassIds;
    for (const dataflow::Lattice<AliasEquivalenceClass> *operand :
         operandLattices)
      llvm::append_range(eqClassIds, operand->getValue().getEqClassIds());
    propagateIfChanged(results[0],
                       results[0]->join(AliasEquivalenceClass(eqClassIds)));
    return success();
  }

  // SplitRegisterRangeOp: each result contains all alias classes from operand.
  // This is the inverse of MakeRegisterRangeOp: all results alias together
  // through the common register_range operand.
  if (isa<amdgcn::SplitRegisterRangeOp>(op)) {
    const auto &eqClassIds = operandLattices[0]->getValue().getEqClassIds();
    AliasEquivalenceClass operandClasses(AliasEquivalenceClass::EqClassList(
        eqClassIds.begin(), eqClassIds.end()));
    for (auto result : results)
      propagateIfChanged(result, result->join(operandClasses));
    return success();
  }

  // SelectOp: result aliases the dst operand (DPS semantics).
  // SelectOp is not InstOpInterface (due to separate i1 behavior) but has a dst
  // operand that provides storage.
  if (auto selectOp = dyn_cast<lsir::SelectOp>(op)) {
    // operandLattices[0] is the dst operand.
    const auto &eqClassIds = operandLattices[0]->getValue().getEqClassIds();
    propagateIfChanged(
        results[0], results[0]->join(AliasEquivalenceClass(
                        AliasEquivalenceClass::EqClassList(eqClassIds.begin(),
                                                           eqClassIds.end()))));
    return success();
  }

  // For other operations, set results to UNKNOWN (conservative).
  // Note: branches and other operations of interest are handled through
  // provenance analysis (used by getOrCreateEqClassId).
  for (dataflow::Lattice<AliasEquivalenceClass> *result : results)
    propagateIfChanged(result,
                       result->join(AliasEquivalenceClass::getUnknown()));
  return success();
}

void DPSAliasAnalysis::setToEntryState(
    dataflow::Lattice<AliasEquivalenceClass> *lattice) {
  // Initialize block arguments to uninitialized (bottom of lattice). They will
  // receive concrete equivalence class IDs via dataflow from incoming edges
  // (cf.br, cf.cond_br). If conflicting information arrives, only then will the
  // join produce CONFLICT.
  propagateIfChanged(lattice,
                     lattice->join(AliasEquivalenceClass::getUninitialized()));
}

EqClassID DPSAliasAnalysis::getOrCreateEqClassId(Value alloca) {
  if (auto it = valueToEqClassIdMap.find(alloca);
      it != valueToEqClassIdMap.end())
    return it->second;

  // Without provenance analysis, just create a new ID (no phi-coalescing).
  if (!provenanceAnalysis) {
    EqClassID eqClassId = idsToValuesMap.size();
    valueToEqClassIdMap[alloca] = eqClassId;
    idsToValuesMap.push_back(alloca);
    return eqClassId;
  }

  // Phi-equivalence implies alias equivalence.
  FailureOr<Value> maybeEquivalentAlloca =
      provenanceAnalysis->getCanonicalPhiEquivalentAlloca(alloca);

  //  No different equivalent alloca - create new ID.
  if (failed(maybeEquivalentAlloca) || *maybeEquivalentAlloca == alloca) {
    EqClassID eqClassId = idsToValuesMap.size();
    valueToEqClassIdMap[alloca] = eqClassId;
    idsToValuesMap.push_back(alloca);
    return eqClassId;
  }

  // Different equivalent alloca - getOrCreate its DPS alias class ID.
  // Note: recursive call to getOrCreateEqClassId is guaranteed to succeed now.
  EqClassID eqClassId = getOrCreateEqClassId(*maybeEquivalentAlloca);
  valueToEqClassIdMap[alloca] = eqClassId;

  return eqClassId;
}
