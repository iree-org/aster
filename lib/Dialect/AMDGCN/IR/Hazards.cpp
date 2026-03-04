//===- Hazards.cpp - AMDGCN hazard detection ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if two register types overlap.
static bool checkOverlap(AMDGCNRegisterTypeInterface lhs,
                         AMDGCNRegisterTypeInterface rhs) {
  if (!lhs || !rhs || !lhs.hasAllocatedSemantics() ||
      !rhs.hasAllocatedSemantics())
    return false;

  // If the register kinds are different, they cannot overlap.
  if (lhs.getRegisterKind() != rhs.getRegisterKind())
    return false;

  int16_t lhsBegin = lhs.getAsRange().begin().getRegister();
  int16_t lhsEnd = lhsBegin + lhs.getAsRange().size();
  int16_t rhsBegin = rhs.getAsRange().begin().getRegister();
  int16_t rhsEnd = rhsBegin + rhs.getAsRange().size();

  // Check if the ranges overlap.
  return lhsEnd > rhsBegin && rhsEnd > lhsBegin;
}

//===----------------------------------------------------------------------===//
// Hazard
//===----------------------------------------------------------------------===//

bool Hazard::compare(const Hazard &other, DominanceInfo &domInfo) const {
  auto cmpAttr = [](HazardAttrInterface lhs, HazardAttrInterface rhs) {
    // If they are the same, return false because this is checking `<`.
    if (lhs == rhs)
      return false;
    return lhs.getAbstractAttribute().getName() <
           rhs.getAbstractAttribute().getName();
  };

  InstCounts lhsCounts = getInstCounts();
  InstCounts rhsCounts = other.getInstCounts();
  // 1. Lower nop counts first.
  if (!(lhsCounts == rhsCounts))
    return lhsCounts < rhsCounts;

  Operation *opA = getOp();
  Operation *opB = other.getOp();

  // 2. If the operations are the same, sort by operand number and hazard.
  if (opA == opB) {
    int32_t operandA = getOperand() ? getOperand()->getOperandNumber() : -1;
    int32_t operandB =
        other.getOperand() ? other.getOperand()->getOperandNumber() : -1;
    if (operandA != operandB)
      return operandA < operandB;
    return cmpAttr(getHazard(), other.getHazard());
  }

  // 3. Dominance: dominated operation comes first.
  if (domInfo.properlyDominates(opA, opB))
    return true;
  if (domInfo.properlyDominates(opB, opA))
    return false;

  // 4. Tiebreaker: hazard.
  return cmpAttr(getHazard(), other.getHazard());
}

//===----------------------------------------------------------------------===//
// CDNA3 Hazards
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Hazard attributes
//===----------------------------------------------------------------------===//

bool CDNA3StoreHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || !metadata->hasProp(InstProp::IsValu))
    return false;

  auto storeOp = cast<StoreOp>(hazard.getOp());
  AMDGCNRegisterTypeInterface writeRegTy = storeOp.getData().getType();

  assert(writeRegTy.hasAllocatedSemantics() &&
         "Write register type must have allocated semantics");

  // Check if the VALU instruction writes to the same register as the store.
  return llvm::any_of(TypeRange(instOp.getInstOuts()), [&](Type out) {
    return checkOverlap(dyn_cast<AMDGCNRegisterTypeInterface>(out), writeRegTy);
  });
}

void CDNA3StoreHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  auto storeOp = dyn_cast<StoreOp>(instOp.getOperation());
  if (!storeOp)
    return;

  // Check if it has allocated semantics.
  RegisterTypeInterface regTy = storeOp.getData().getType();
  if (!regTy || !regTy.hasAllocatedSemantics())
    return;

  const InstMetadata *metadata = storeOp.getInstMetadata();
  if (!metadata)
    return;

  // Handle buffer ops.
  if (llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         metadata->getOpCode())) {
    if (!storeOp.getDynamicOffset()) {
      hazards.push_back(Hazard(
          *this, storeOp.getDataMutable(),
          InstCounts(/*v_nops=*/requiredWaits, /*s_nops=*/0, /*ds_nops=*/0)));
    }
  }

  // Handle global ops.
  if (metadata->hasProp(InstProp::Global)) {
    hazards.push_back(Hazard(
        *this, storeOp.getDataMutable(),
        InstCounts(/*v_nops=*/requiredWaits, /*s_nops=*/0, /*ds_nops=*/0)));
  }
}

//===----------------------------------------------------------------------===//
// Hazard manager implementation
//===----------------------------------------------------------------------===//

LogicalResult CDNA3Hazards::getHazards(AMDGCNInstOpInterface instOp,
                                       SmallVectorImpl<Hazard> &hazards) {
  CDNA3StoreHazardAttr::get(instOp.getContext())
      .populateHazardsFor(instOp, hazards);
  return success();
}
