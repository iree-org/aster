//===- ConstantMemoryOffsetAnalysis.cpp - Constant offset analysis ===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ConstantMemoryOffsetAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/ValueOrConst.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

#define DEBUG_TYPE "constant-memory-offset-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetLattice
//===----------------------------------------------------------------------===//

void ConstantMemoryOffsetLattice::print(raw_ostream &os) const {
  if (isTop()) {
    os << "<top>";
    return;
  }

  os << "{ values: [";
  bool first = true;
  for (const auto &pair : constantOffsetValues) {
    // Only print entries with non-zero constant offset
    if (pair.second.constantOffset == 0)
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << pair.first << ": { constant_offset: " << pair.second.constantOffset;
    os << ", base_value: ";
    if (pair.second.hasBaseValue()) {
      pair.second.baseValue->print(os);
    } else {
      os << "none";
    }
    os << " }";
  }
  os << "] }";
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::ConstantMemoryOffsetLattice)

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//
static ConstantMemoryOffsetInfo
extractFromAffineApply(affine::AffineApplyOp affineApply) {
  AffineMap map = affineApply.getMap();
  ValueRange mapOperandsRange = affineApply.getMapOperands();
  SmallVector<Value> mapOperands(mapOperandsRange.begin(),
                                 mapOperandsRange.end());

  // Check if the affine map has a single result expression
  if (map.getNumResults() != 1)
    return ConstantMemoryOffsetInfo(0, map, mapOperands);

  AffineExpr expr = map.getResult(0);

  // Not in the form "constant + expr", return original map with constant = 0
  if (expr.getKind() != AffineExprKind::Add) {
    return ConstantMemoryOffsetInfo(0, map, mapOperands);
  }

  // Check if the expression is of the form "constant + expr" or "expr +
  // constant"
  auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!addExpr)
    return ConstantMemoryOffsetInfo(0, map, mapOperands);

  assert(!isa<AffineConstantExpr>(addExpr.getLHS()) &&
         "Affine expression should be canonicalized with constants on the RHS");

  // Affine expressions are canonicalized with constants on the RHS.
  if (auto constExpr = dyn_cast<AffineConstantExpr>(addExpr.getRHS())) {
    int64_t constantOffset = constExpr.getValue();
    // Create a new affine map with just the non-constant part
    AffineMap newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                      {addExpr.getLHS()}, map.getContext());
    return ConstantMemoryOffsetInfo(constantOffset, newMap, mapOperands);
  }

  return ConstantMemoryOffsetInfo(0, map, mapOperands);
}

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetAnalysis - Helper functions
//===----------------------------------------------------------------------===//

bool ConstantMemoryOffsetAnalysis::handleTopPropagation(
    const ConstantMemoryOffsetLattice &before,
    ConstantMemoryOffsetLattice *after) {
  if (before.isTop() || after->isTop()) {
    propagateIfChanged(after, after->setToTop());
    return true;
  }
  return false;
}

#define DUMP_STATE_HELPER(name, obj)                                           \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS(                                                                   \
        [&](raw_ostream &os) { os << "Visiting " name ": " << obj << "\n"; }); \
  });

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetAnalysis - Visit methods
//===----------------------------------------------------------------------===//

LogicalResult ConstantMemoryOffsetAnalysis::visitOperation(
    Operation *op, const ConstantMemoryOffsetLattice &before,
    ConstantMemoryOffsetLattice *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()));

  if (handleTopPropagation(before, after))
    return success();

  ChangeResult result = ChangeResult::NoChange;

  // Propagate the state from before to after.
  const auto &beforeInfo = before.getValueInfo();
  for (const auto &pair : beforeInfo) {
    result |= after->setInfo(pair.first, pair.second);
  }

  // Handle affine.apply operations: extract the constant offset and the base
  // value.
  if (auto affineApply = dyn_cast<affine::AffineApplyOp>(op)) {
    assert(affineApply->getNumResults() == 1 &&
           "affine.apply must have one result");
    ConstantMemoryOffsetInfo info = extractFromAffineApply(affineApply);
    result |= after->setInfo(affineApply->getResult(0), info);
    propagateIfChanged(after, result);
    return success();
  }

  // For casts, just propagate the previous value.
  if (auto indexCast = dyn_cast<CastOpInterface>(op)) {
    // Operand is an input, so get its info from the before state
    auto info = before.getInfo(indexCast->getOperand(0));
    result |= after->setInfo(indexCast->getResult(0), info);
    propagateIfChanged(after, result);
    return success();
  }
  if (auto toReg = dyn_cast<lsir::ToRegOp>(op)) {
    // Operand is an input, so get its info from the before state
    auto info = before.getInfo(toReg->getOperand(0));
    result |= after->setInfo(toReg->getResult(0), info);
    propagateIfChanged(after, result);
    return success();
  }

  // For v_mov_b32_e32, just propagate the value from src0 (input operand)
  if (auto vMovB32E32 = dyn_cast<amdgcn::inst::VOP1Op>(op)) {
    if (vMovB32E32.getOpcode() == amdgcn::OpCode::V_MOV_B32_E32) {
      // src0 is an input operand, so get its info from the before state
      auto info = before.getInfo(vMovB32E32.getSrc0());
      result |= after->setInfo(vMovB32E32.getResult(), info);
      propagateIfChanged(after, result);
      return success();
    }
  }
  propagateIfChanged(after, result);
  return success();
}

void ConstantMemoryOffsetAnalysis::visitBlockTransfer(
    Block *block, ProgramPoint *point, Block *predecessor,
    const ConstantMemoryOffsetLattice &before,
    ConstantMemoryOffsetLattice *after) {
  DUMP_STATE_HELPER("block", block);

  assert(false && "block transfer not supported atm");
  return;
}

void ConstantMemoryOffsetAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const ConstantMemoryOffsetLattice &before,
    ConstantMemoryOffsetLattice *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()));

  assert(false && "call control flow transfer not supported atm");
}

void ConstantMemoryOffsetAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const ConstantMemoryOffsetLattice &before,
    ConstantMemoryOffsetLattice *after) {
  DUMP_STATE_HELPER("branch op",
                    OpWithFlags(branch, OpPrintingFlags().skipRegions()));

  assert(false && "region branch control flow transfer not supported atm");
}

void ConstantMemoryOffsetAnalysis::setToEntryState(
    ConstantMemoryOffsetLattice *lattice) {
  // At entry, everything is empty.
  // Still, mark it as changed so the forward analysis kicks off.
  propagateIfChanged(lattice, ChangeResult::Change);
}

#undef DUMP_STATE_HELPER
