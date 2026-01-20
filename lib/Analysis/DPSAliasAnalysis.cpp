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
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>

#define DEBUG_TYPE "dps-alias-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Variable
//===----------------------------------------------------------------------===//

void Variable::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<TOP>";
    return;
  }
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  os << "[";
  llvm::interleaveComma(*variableIds, os);
  os << "]";
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Variable &var) {
  var.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// DPSAliasAnalysis
//===----------------------------------------------------------------------===//

/// Helper to mark the IR as ill-formed if any of the given lattices is top.
/// Returns true if any lattice is top.
static bool isIllFormed(bool &illFormed,
                        ArrayRef<const dataflow::Lattice<Variable> *> lattices,
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
                          ArrayRef<dataflow::Lattice<Variable> *> lattices,
                          ValueRange results) {
  for (auto [result, lattice] : llvm::zip_equal(results, lattices)) {
    if (lattice->getValue().isTop() &&
        isa<RegisterTypeInterface>(result.getType())) {
      illFormed = true;
    }
  }
}

LogicalResult DPSAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<Variable> *> operandLattices,
    ArrayRef<dataflow::Lattice<Variable> *> results) {
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
  bool isIllFormedOperand = isIllFormed(illFormed, operandLattices, op->getOperands());
  if (isIllFormedOperand) {
    for (dataflow::Lattice<Variable> *result : results)
      propagateIfChanged(result, result->join(Variable::getTop()));
    return success();
  }

  // Handle specific operations.
  // Each AllocaOp defines a new variable.
  if (auto aOp = dyn_cast<AllocaOp>(op)) {
    // For AllocaOp, we can assign a unique variable ID
    int32_t varId = valueToVarIdMap.size();
    valueToVarIdMap[aOp.getResult()] = varId;
    idsToValuesMap.push_back(aOp.getResult());
    assert(idsToValuesMap.size() == valueToVarIdMap.size() &&
           "idsToValuesMap and valueToVarIdMap size mismatch");
    propagateIfChanged(results[0], results[0]->join(Variable({varId})));
    return success();
  }

  // Handle InstOpInterface operations.
  // Each InstOpInterface ties its results to variables tied to the matching DPS operand.
  if (auto instOp = dyn_cast<InstOpInterface>(op)) {
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      size_t idx = operand.getOperandNumber();
      propagateIfChanged(results[idx],
                         results[idx]->join(operandLattices[idx]->getValue()));
    }
    return success();
  }

  // Handle MakeRegisterRangeOp operations.
  // Each MakeRegisterRangeOp ties its result to variables tied to all the operandLattices.
  if (auto mOp = dyn_cast<amdgcn::MakeRegisterRangeOp>(op)) {
    Variable::VIdList varIds;
    for (const dataflow::Lattice<Variable> *operand : operandLattices)
      llvm::append_range(varIds, operand->getValue().getVariableIds());
    propagateIfChanged(results[0], results[0]->join(Variable(varIds)));
    return success();
  }

  // Handle SplitRegisterRangeOp operations.
  // Each SplitRegisterRangeOp ties its results to variables tied to all the
  // variables tied to the unique operand.
  if (isa<amdgcn::SplitRegisterRangeOp>(op)) {
    for (auto [idx, result, vid] :
         llvm::enumerate(results, operandLattices[0]->getValue().getVariableIds())) {
      propagateIfChanged(result, result->join(Variable({vid})));
    }
    return success();
  }

  // For other operations, we conservatively set all results to top.
  setAllToEntryStates(results);
  return success();
}

void DPSAliasAnalysis::setToEntryState(dataflow::Lattice<Variable> *lattice) {
  // Set the lattice to top (overdefined) at entry points
  propagateIfChanged(lattice, lattice->join(Variable::getTop()));
}
