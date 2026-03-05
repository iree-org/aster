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
#include "aster/Dialect/AMDGCN/IR/HazardManager.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-hazards"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static MutableArrayRef<OpOperand> getOpOperands(OperandRange operands) {
  if (operands.empty())
    return {};
  return MutableArrayRef<OpOperand>(operands.getBase(), operands.size());
}

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

/// Check if a register kind is VCC or EXEC (written by first instruction).
static bool isVccOrExecKind(RegisterKind kind) {
  return llvm::is_contained({RegisterKind::VCC, RegisterKind::VCC_LO,
                             RegisterKind::VCC_HI, RegisterKind::EXEC,
                             RegisterKind::EXEC_LO, RegisterKind::EXEC_HI},
                            kind);
}

/// Check if a register kind is VCCZ or EXECZ (read by second instruction).
static bool isVcczOrExeczKind(RegisterKind kind) {
  return kind == RegisterKind::VCCZ || kind == RegisterKind::EXECZ;
}

/// Check if the instruction supports the given ISA version.
static bool instSupportsIsa(const InstMetadata *md, ISAVersion isaVer) {
  if (!md)
    return false;
  ArrayRef<ISAVersion> isas = md->getISAVersions();
  if (isas.empty())
    return true; // Available on all targets.
  return llvm::is_contained(isas, isaVer);
}

/// Check if a VMEM instruction reads from SGPRs overlapping the given type.
static bool vmemReadsFromSgprType(AMDGCNInstOpInterface instOp,
                                  AMDGCNRegisterTypeInterface targetRegTy) {
  if (!targetRegTy || targetRegTy.getRegisterKind() != RegisterKind::SGPR)
    return false;

  for (Value input : instOp.getInstIns()) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    if (inputRegTy && inputRegTy.getRegisterKind() == RegisterKind::SGPR &&
        checkOverlap(inputRegTy, targetRegTy))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Hazard
//===----------------------------------------------------------------------===//

bool Hazard::compare(const Hazard &other, DominanceInfo &domInfo) const {
  auto cmpAttr = [](HazardCheckerAttrInterface lhs,
                    HazardCheckerAttrInterface rhs) {
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
// Case 1: SetRegGetRegHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetRegGetRegHazardAttr::matchInst(const InstMetadata *,
                                            ISAVersion) const {
  return false; // TODO: S_SETREG and S_GETREG not implemented.
}

void CDNA3SetRegGetRegHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SetRegGetRegHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 2: SetRegSetRegHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetRegSetRegHazardAttr::matchInst(const InstMetadata *,
                                            ISAVersion) const {
  return false; // TODO: S_SETREG not implemented.
}

void CDNA3SetRegSetRegHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SetRegSetRegHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 3: SetVskipGetRegHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetVskipGetRegHazardAttr::matchInst(const InstMetadata *,
                                              ISAVersion) const {
  return false; // TODO: SET_VSKIP and S_GETREG not implemented.
}

void CDNA3SetVskipGetRegHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SetVskipGetRegHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 4: SetRegVskipVectorHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetRegVskipVectorHazardAttr::matchInst(const InstMetadata *,
                                                 ISAVersion) const {
  return false; // TODO: S_SETREG not implemented.
}

void CDNA3SetRegVskipVectorHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SetRegVskipVectorHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 5: VccExecVcczExeczHazard
//===----------------------------------------------------------------------===//
bool CDNA3VccExecVcczExeczHazardAttr::matchInst(const InstMetadata *md,
                                                ISAVersion isaVer) const {
  return md && instSupportsIsa(md, isaVer) && md->hasProp(InstProp::IsValu);
}

void CDNA3VccExecVcczExeczHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || !metadata->hasProp(InstProp::IsValu))
    return;

  // Check if any output writes VCC or EXEC.
  bool writesVccOrExec = llvm::any_of(instOp.getInstOuts(), [](Value out) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(out.getType());
    return regTy && isVccOrExecKind(regTy.getRegisterKind());
  });
  if (!writesVccOrExec)
    return;

  hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                           instOp.getOperation(), getInstCounts(0)));
}
bool CDNA3VccExecVcczExeczHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || !metadata->hasProp(InstProp::IsValu))
    return false;

  // Check if any input reads VCCZ or EXECZ as data source.
  return llvm::any_of(instOp.getInstIns(), [](Value input) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    return regTy && isVcczOrExeczKind(regTy.getRegisterKind());
  });
}

//===----------------------------------------------------------------------===//
// Case 6: ValuSgprReadlaneHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuSgprReadlaneHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: V_READLANE, V_WRITELANE not implemented.
}

void CDNA3ValuSgprReadlaneHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3ValuSgprReadlaneHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 7: VccDivFmasHazard
//===----------------------------------------------------------------------===//
bool CDNA3VccDivFmasHazardAttr::matchInst(const InstMetadata *,
                                          ISAVersion) const {
  return false; // TODO: V_DIV_FMAS not implemented.
}

void CDNA3VccDivFmasHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3VccDivFmasHazardAttr::isHazardTriggered(const Hazard &,
                                                  AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 8: StoreWriteDataHazard
//===----------------------------------------------------------------------===//

bool CDNA3StoreWriteDataHazardAttr::matchInst(const InstMetadata *md,
                                              ISAVersion isaVer) const {
  if (!md || !instSupportsIsa(md, isaVer))
    return false;
  OpCode op = md->getOpCode();
  if (md->hasProp(InstProp::Global))
    return true;
  if (md->hasProp(InstProp::Buffer) &&
      llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         op)) {
    return true;
  }
  return false;
}

void CDNA3StoreWriteDataHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  auto storeOp = dyn_cast<StoreOp>(instOp.getOperation());
  if (!storeOp)
    return;
  RegisterTypeInterface regTy = storeOp.getData().getType();
  if (!regTy)
    return;
  const InstMetadata *metadata = storeOp.getInstMetadata();
  if (!metadata)
    return;

  if (llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         metadata->getOpCode())) {
    // If the dynamic offset is not set, there is no hazard.
    if (!storeOp.getDynamicOffset())
      return;
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             storeOp.getDataMutable(), getInstCounts(0)));
    return;
  }
  if (metadata->hasProp(InstProp::Global)) {
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             storeOp.getDataMutable(), getInstCounts(0)));
    return;
  }
  // TODO: FLAT_ATOMIC_[F]CMPSWAP_X2, BUFFER_STORE_FORMAT_XYZ/XYZW,
  // BUFFER_ATOMIC_[F]CMPSWAP_X2 not implemented in AMDGCN.
}

bool CDNA3StoreWriteDataHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");
  // Case 8: Any instruction that writes to writedata VGPRs (non-VALU gets 1
  // wait). Case 9 (CDNA3StoreHazard) handles VALU with 2 waits.
  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || metadata->hasProp(InstProp::IsValu))
    return false; // VALU is handled by CDNA3StoreHazard

  auto storeOp = cast<StoreOp>(hazard.getOp());
  AMDGCNRegisterTypeInterface writeRegTy = storeOp.getData().getType();
  if (!writeRegTy.hasAllocatedSemantics())
    return false;

  return llvm::any_of(TypeRange(instOp.getInstOuts()), [&](Type out) {
    return checkOverlap(dyn_cast<AMDGCNRegisterTypeInterface>(out), writeRegTy);
  });
}

//===----------------------------------------------------------------------===//
// Case 9: StoreHazard
//===----------------------------------------------------------------------===//

bool CDNA3StoreHazardAttr::matchInst(const InstMetadata *md,
                                     ISAVersion isaVer) const {
  if (!md || !instSupportsIsa(md, isaVer))
    return false;
  OpCode op = md->getOpCode();
  if (md->hasProp(InstProp::Global))
    return true;
  if (md->hasProp(InstProp::Buffer) &&
      llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         op)) {
    return true;
  }
  return false;
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

    // If the dynamic offset is not set, there is no hazard.
    if (!storeOp.getDynamicOffset())
      return;
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             storeOp.getDataMutable(), getInstCounts(0)));
    return;
  }

  // Handle global ops.
  if (metadata->hasProp(InstProp::Global)) {
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             storeOp.getDataMutable(), getInstCounts(0)));
    return;
  }
  // TODO: FLAT_ATOMIC_[F]CMPSWAP_X2, BUFFER_STORE_FORMAT_XYZ/XYZW,
  // BUFFER_ATOMIC_[F]CMPSWAP_X2 not implemented in AMDGCN.
}

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

//===----------------------------------------------------------------------===//
// Case 10: ValuSgprVmemHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuSgprVmemHazardAttr::matchInst(const InstMetadata *md,
                                            ISAVersion isaVer) const {
  return md && instSupportsIsa(md, isaVer) && md->hasProp(InstProp::IsValu);
}

void CDNA3ValuSgprVmemHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || !metadata->hasProp(InstProp::IsValu))
    return;

  for (OpOperand &operand : getOpOperands(instOp.getInstOuts())) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(operand.get().getType());
    if (!regTy || regTy.getRegisterKind() != RegisterKind::SGPR)
      continue;

    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this), operand,
                             getInstCounts(0)));
  }
}

bool CDNA3ValuSgprVmemHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  const InstMetadata *metadata = instOp.getInstMetadata();
  if (!metadata || !metadata->hasProp(InstProp::IsVmem))
    return false;

  OpOperand *sgprOperand = hazard.getOperand();
  if (!sgprOperand)
    return false;

  auto sgprRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(sgprOperand->get().getType());
  if (!sgprRegTy || sgprRegTy.getRegisterKind() != RegisterKind::SGPR)
    return false;

  return vmemReadsFromSgprType(instOp, sgprRegTy);
}

//===----------------------------------------------------------------------===//
// Case 11: SaluM0GdsSendmsgHazard
//===----------------------------------------------------------------------===//
bool CDNA3SaluM0GdsSendmsgHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: GDS and S_SENDMSG not implemented.
}

void CDNA3SaluM0GdsSendmsgHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SaluM0GdsSendmsgHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 12: ValuVgprDppHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuVgprDppHazardAttr::matchInst(const InstMetadata *,
                                           ISAVersion) const {
  return false; // TODO: DPP modifier not implemented.
}

void CDNA3ValuVgprDppHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3ValuVgprDppHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 13: ExecDppHazard
//===----------------------------------------------------------------------===//
bool CDNA3ExecDppHazardAttr::matchInst(const InstMetadata *, ISAVersion) const {
  return false; // TODO: DPP modifier not implemented.
}

void CDNA3ExecDppHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3ExecDppHazardAttr::isHazardTriggered(const Hazard &,
                                               AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 14: MixedVccConstantHazard
//===----------------------------------------------------------------------===//
bool CDNA3MixedVccConstantHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: VCC/SGPR alias tracking not implemented.
}

void CDNA3MixedVccConstantHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3MixedVccConstantHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 15: SetRegTrapstsRfeHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetRegTrapstsRfeHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: S_SETREG, RFE, RFE_restore not implemented.
}

void CDNA3SetRegTrapstsRfeHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SetRegTrapstsRfeHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 16: SaluM0LdsHazard
//===----------------------------------------------------------------------===//
bool CDNA3SaluM0LdsHazardAttr::matchInst(const InstMetadata *,
                                         ISAVersion) const {
  return false; // TODO: LDS add-TID, buffer_store_LDS not implemented.
}

void CDNA3SaluM0LdsHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SaluM0LdsHazardAttr::isHazardTriggered(const Hazard &,
                                                 AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 17: SaluM0MoverelHazard
//===----------------------------------------------------------------------===//
bool CDNA3SaluM0MoverelHazardAttr::matchInst(const InstMetadata *,
                                             ISAVersion) const {
  return false; // TODO: S_MOVEREL not implemented.
}

void CDNA3SaluM0MoverelHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SaluM0MoverelHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 18: ValuSgprValuReadHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuSgprValuReadHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: SGPR/VCC dependency tracking not implemented.
}

void CDNA3ValuSgprValuReadHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3ValuSgprValuReadHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 19: ValuVgprReadlaneHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuVgprReadlaneHazardAttr::matchInst(const InstMetadata *,
                                                ISAVersion) const {
  return false; // TODO: V_READLANE not implemented.
}

void CDNA3ValuVgprReadlaneHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3ValuVgprReadlaneHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 20: OpselSdwaHazard
//===----------------------------------------------------------------------===//
bool CDNA3OpselSdwaHazardAttr::matchInst(const InstMetadata *,
                                         ISAVersion) const {
  return false; // TODO: OPSEL and SDWA modifiers not implemented.
}

void CDNA3OpselSdwaHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3OpselSdwaHazardAttr::isHazardTriggered(const Hazard &,
                                                 AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case 21: TransOpHazard
//===----------------------------------------------------------------------===//
bool CDNA3TransOpHazardAttr::matchInst(const InstMetadata *, ISAVersion) const {
  return false; // TODO: Trans op detection not implemented.
}

void CDNA3TransOpHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3TransOpHazardAttr::isHazardTriggered(const Hazard &,
                                               AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Hazard manager implementation
//===----------------------------------------------------------------------===//

void HazardManager::getHazardRaisersFor(
    ISAVersion version,
    SmallVectorImpl<HazardRaiserAttrInterface> &hazardRaisers) {
  hazardRaisers.clear();
  if (version == ISAVersion::CDNA3) {
    MLIRContext *ctx = getTopOp()->getContext();
    hazardRaisers.push_back(CDNA3StoreHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3StoreWriteDataHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3VccExecVcczExeczHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3ValuSgprVmemHazardAttr::get(ctx));
  }
}

void HazardManager::populateHazardsFor(ISAVersion version) {
  hazardAttrs.clear();
  opcodeToHazardAttrs.clear();

  // Collect all hazard raisers for the given ISA version.
  SmallVector<HazardRaiserAttrInterface> hazardRaisers;
  getHazardRaisersFor(version, hazardRaisers);

  // Since we are matching by opcode, we don't need to visit the same opcode
  // twice.
  llvm::SmallDenseSet<OpCode> seen;

  // Walk the top operation and populate the hazard raisers by opcode.
  getTopOp()->walk([&](AMDGCNInstOpInterface instOp) {
    const InstMetadata *md = instOp.getInstMetadata();

    // Skip instructions that don't have metadata or have already been visited.
    if (!md || !seen.insert(md->getOpCode()).second)
      return;

    // Check if the instruction has a hazard raiser.
    OpCode opcode = md->getOpCode();
    for (HazardRaiserAttrInterface hazardRaiser : hazardRaisers) {
      if (!hazardRaiser.matchInst(md, version))
        continue;
      hazardAttrs.push_back({opcode, hazardRaiser});
    }
  });

  // Sort the hazard raisers by opcode and attribute pointer.
  llvm::sort(hazardAttrs,
             [](const std::pair<OpCode, HazardRaiserAttrInterface> &a,
                const std::pair<OpCode, HazardRaiserAttrInterface> &b) {
               return std::make_pair(a.first, a.second.getAsOpaquePointer()) <
                      std::make_pair(b.first, b.second.getAsOpaquePointer());
             });

  // Populate the opcode to hazard attrs map.
  for (int32_t i = 0, e = hazardAttrs.size(); i < e;) {
    OpCode opcode = hazardAttrs[i].first;
    int32_t start = i;
    while (i < e && hazardAttrs[i].first == opcode)
      ++i;
    opcodeToHazardAttrs[opcode] = {start, i};
  }

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Hazards patterns for " << stringifyISAVersion(version) << ":\n";
    for (auto [opcode, hazardRaiser] : hazardAttrs) {
      os << "  " << stringifyOpCode(opcode) << ": " << hazardRaiser << "\n";
    }
  });
}

LogicalResult HazardManager::getHazards(AMDGCNInstOpInterface instOp,
                                        SmallVectorImpl<Hazard> &hazards) {
  LDBG() << "Getting hazards for instruction: "
         << OpWithFlags(instOp, OpPrintingFlags().skipRegions());
  const InstMetadata *md = instOp.getInstMetadata();
  if (!md) {
    LDBG() << "- no metadata for instruction";
    return success();
  }

  // Lookup if there are any hazard raisers for the given instruction.
  auto [start, end] = opcodeToHazardAttrs.lookup_or(md->getOpCode(), {-1, -1});
  if (start == -1 || end == -1) {
    LDBG() << "- no hazard patterns for instruction";
    return success();
  }

  // Check if all output and input operands have allocated semantics, bail if
  // not.
  auto isAllocType = [](Type out) {
    auto regTy = dyn_cast<RegisterTypeInterface>(out);
    return regTy && regTy.hasAllocatedSemantics();
  };

  if (!llvm::all_of(TypeRange(instOp.getInstOuts()), isAllocType))
    return instOp.emitError("output operands must have allocated semantics");
  if (!llvm::all_of(TypeRange(instOp.getInstIns()), isAllocType))
    return instOp.emitError("input operands must have allocated semantics");

  // Populate the hazards for the given instruction.
  for (int32_t i = start; i < end; ++i)
    hazardAttrs[i].second.populateHazardsFor(instOp, hazards);

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "- Collected hazards: ";
    for (const Hazard &hazard : hazards)
      os << " - " << hazard << "\n";
  });
  return success();
}
