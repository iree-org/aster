//===- AllocaAliasAnalysis.cpp - Alloca alias analysis --------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/AllocaAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-alloca-alias-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// AllocaAlias
//===----------------------------------------------------------------------===//

void AllocaAlias::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<TOP>";
    return;
  }
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  os << "[";
  llvm::interleaveComma(*allocaIds, os);
  os << "]";
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AllocaAlias &alias) {
  alias.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// AllocaAliasAnalysis
//===----------------------------------------------------------------------===//

/// Helper to mark the IR as ill-formed if any of the given lattices is top.
/// Returns true if any lattice is top.
static bool
isIllFormed(bool &illFormed,
            ArrayRef<const dataflow::Lattice<AllocaAlias> *> lattices,
            ValueRange operands) {
  if (illFormed)
    return false;
  for (auto [operand, lattice] : llvm::zip_equal(operands, lattices)) {
    if (lattice->getValue().isTop() &&
        isa<RegisterTypeInterface>(operand.getType())) {
      return (illFormed = true);
    }
  }
  return false;
}

static void isIllFormedOp(bool &illFormed,
                          ArrayRef<dataflow::Lattice<AllocaAlias> *> lattices,
                          ValueRange results) {
  for (auto [result, lattice] : llvm::zip_equal(results, lattices)) {
    if (lattice->getValue().isTop() &&
        isa<RegisterTypeInterface>(result.getType())) {
      illFormed = true;
    }
  }
}

LogicalResult AllocaAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AllocaAlias> *> operands,
    ArrayRef<dataflow::Lattice<AllocaAlias> *> results) {
  // Check if the op results are ill-formed.
  auto _atExit = llvm::make_scope_exit([&]() {
    if (ValueRange vals = op->getResults(); !vals.empty())
      isIllFormedOp(illFormed, results, vals);

    // Log the lattices at exit for debugging.
    LDBG_OS([&](llvm::raw_ostream &os) {
      os << "Lattices for op: " << *op << " =>\n  ";
      llvm::interleaveComma(llvm::enumerate(results), os, [&](auto idxLattice) {
        os << idxLattice.index() << ": ";
        idxLattice.value()->getValue().print(os);
      });
    });
  });

  // Early exit if any register-like operand lattice is top.
  bool isIllFormedOperand = isIllFormed(illFormed, operands, op->getOperands());
  if (isIllFormedOperand) {
    for (dataflow::Lattice<AllocaAlias> *result : results)
      propagateIfChanged(result, result->join(AllocaAlias::getTop()));
    return success();
  }

  // Handle specific operations.
  if (auto aOp = dyn_cast<AllocaOp>(op)) {
    // For AllocaOp, we can assign a unique alloca ID
    int32_t allocaId = valueToAllocaIdMap.size();
    valueToAllocaIdMap[aOp.getResult()] = allocaId;
    allocaIdsToValuesMap.push_back(aOp.getResult());
    assert(allocaIdsToValuesMap.size() == valueToAllocaIdMap.size() &&
           "allocaIdsToValuesMap and valueToAllocaIdMap size mismatch");
    propagateIfChanged(results[0], results[0]->join(AllocaAlias({allocaId})));
    return success();
  }

  // Handle InstOpInterface operations.
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      size_t idx = operand.getOperandNumber();
      propagateIfChanged(results[idx],
                         results[idx]->join(operands[idx]->getValue()));
    }
    return success();
  }

  // Handle MakeRegisterRangeOp operations.
  if (auto mOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
    AllocaAlias::IdList allocaIds;
    for (const dataflow::Lattice<AllocaAlias> *operand : operands)
      llvm::append_range(allocaIds, operand->getValue().getAllocaIds());
    propagateIfChanged(results[0], results[0]->join(AllocaAlias(allocaIds)));
    return success();
  }

  // Handle SplitRegisterRangeOp operations.
  if (isa<amdgcn::SplitRegisterRangeOp>(op)) {
    for (auto [idx, result, allocaId] :
         llvm::enumerate(results, operands[0]->getValue().getAllocaIds())) {
      propagateIfChanged(result, result->join(AllocaAlias({allocaId})));
    }
    return success();
  }

  // For other operations, we conservatively set all results to top.
  setAllToEntryStates(results);
  return success();
}

void AllocaAliasAnalysis::setToEntryState(
    dataflow::Lattice<AllocaAlias> *lattice) {
  // Set the lattice to top (overdefined) at entry points
  propagateIfChanged(lattice, lattice->join(AllocaAlias::getTop()));
}
