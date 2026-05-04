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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
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

static int16_t getMfmaPassCase(OpCode op) {
  switch (op) {
  // 16 cycles -> 4 passes (case 1), per Table 37: 16x16x16, 16x16x32
  case OpCode::V_MFMA_F32_16X16X16_F16:
  case OpCode::V_MFMA_F32_16X16X16_BF16:
  case OpCode::V_MFMA_F16_16X16X16_F16:
  case OpCode::V_MFMA_F32_16X16X32_FP8_FP8:
  case OpCode::V_MFMA_F32_16X16X32_FP8_BF8:
  case OpCode::V_MFMA_F32_16X16X32_BF8_FP8:
  case OpCode::V_MFMA_F32_16X16X32_BF8_BF8:
  case OpCode::V_MFMA_F32_16X16X32_F16:
  case OpCode::V_MFMA_F32_16X16X32_BF16:
    return 1; // 4 passes
  // 32 cycles -> 8 passes (case 2), per Table 37: 32x32x8, 32x32x16; 32x32x64
  // scaled
  case OpCode::V_MFMA_F32_32X32X8_F16:
  case OpCode::V_MFMA_F32_32X32X16_F16:
  case OpCode::V_MFMA_F32_32X32X16_BF16:
  case OpCode::V_MFMA_SCALE_F32_32X32X64_F8F6F4:
    return 2; // 8 passes
  // 64 cycles -> 16 passes (case 3), 16x16x128 has 4x K of 16x16x32
  case OpCode::V_MFMA_SCALE_F32_16X16X128_F8F6F4:
    return 3; // 16 passes
  default:
    return -1; // unknown or not implemented
  }
}

/// Get vdst Value from VOP3PMAIOp or VOP3PScaledMAIOp.
static OpOperand *getMaiVdst(Operation *op) {
  if (auto mai = dyn_cast<inst::VOP3PMAIOp>(op))
    return &mai.getVdstMutable();
  if (auto scaledMai = dyn_cast<inst::VOP3PScaledMAIOp>(op))
    return &scaledMai.getVdstMutable();
  return nullptr;
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
// AllHazard
//===----------------------------------------------------------------------===//

bool AllHazardAttr::matchInst(AMDGCNInstOpInterface, ISAVersion) const {
  return true;
}

void AllHazardAttr::populateHazardsFor(AMDGCNInstOpInterface op,
                                       SmallVectorImpl<Hazard> &hazards) const {
  hazards.push_back(
      Hazard(*this, op.getOperation(), InstCounts(getVNops(), getSNops(), 0)));
}

bool AllHazardAttr::isHazardTriggered(const Hazard &,
                                      AMDGCNInstOpInterface) const {
  return true;
}

//===----------------------------------------------------------------------===//
// CDNA3 Hazards
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Case 1: SetRegGetRegHazard
//===----------------------------------------------------------------------===//
bool CDNA3SetRegGetRegHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SetRegSetRegHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SetVskipGetRegHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SetRegVskipVectorHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3VccExecVcczExeczHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                                ISAVersion isaVer) const {
  return instOp.supportsISA(isaVer) && instOp.hasProp(InstProp::IsValu);
}

void CDNA3VccExecVcczExeczHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!instOp.hasProp(InstProp::IsValu))
    return;

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

  if (!instOp.hasProp(InstProp::IsValu))
    return false;

  return llvm::any_of(instOp.getInstIns(), [](Value input) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    return regTy && isVcczOrExeczKind(regTy.getRegisterKind());
  });
}

//===----------------------------------------------------------------------===//
// Case 6: ValuSgprReadlaneHazard
//===----------------------------------------------------------------------===//
bool CDNA3ValuSgprReadlaneHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3VccDivFmasHazardAttr::matchInst(AMDGCNInstOpInterface,
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

bool CDNA3StoreWriteDataHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                              ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  OpCode op = instOp.getOpCode();
  if (instOp.hasProp(InstProp::Global))
    return true;
  if (instOp.hasProp(InstProp::Buffer) &&
      llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         op))
    return true;
  return false;
}

void CDNA3StoreWriteDataHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  auto storeOp = dyn_cast<StoreOpInterface>(instOp.getOperation());
  if (!storeOp)
    return;
  RegisterTypeInterface regTy =
      dyn_cast<RegisterTypeInterface>(storeOp.getDataOperand().getType());
  if (!regTy)
    return;
  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  if (llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         instOp.getOpCode())) {
    // If the dynamic offset is not set, there is no hazard.
    if (!storeOp.getDynamicOff())
      return;
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             *storeOp.getDataOperand().get(),
                             getInstCounts(0)));
    return;
  }
  if (instOp.hasProp(InstProp::Global)) {
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             *storeOp.getDataOperand().get(),
                             getInstCounts(0)));
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
  if (instOp.getOpCode() == OpCode::Invalid || instOp.hasProp(InstProp::IsValu))
    return false; // VALU is handled by CDNA3StoreHazard

  auto storeOp = cast<StoreOpInterface>(hazard.getOp());
  AMDGCNRegisterTypeInterface writeRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(storeOp.getDataOperand().getType());
  if (!writeRegTy || !writeRegTy.hasAllocatedSemantics())
    return false;

  return llvm::any_of(TypeRange(instOp.getInstOuts()), [&](Type out) {
    return checkOverlap(dyn_cast<AMDGCNRegisterTypeInterface>(out), writeRegTy);
  });
}

//===----------------------------------------------------------------------===//
// Case 9: StoreHazard
//===----------------------------------------------------------------------===//

bool CDNA3StoreHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                     ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  OpCode op = instOp.getOpCode();
  if (instOp.hasProp(InstProp::Global))
    return true;
  if (instOp.hasProp(InstProp::Buffer) &&
      llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         op))
    return true;
  return false;
}

void CDNA3StoreHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  auto storeOp = dyn_cast<StoreOpInterface>(instOp.getOperation());
  if (!storeOp)
    return;

  // Check if it has allocated semantics.
  RegisterTypeInterface regTy =
      dyn_cast<RegisterTypeInterface>(storeOp.getDataOperand().getType());
  if (!regTy || !regTy.hasAllocatedSemantics())
    return;

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  // Handle buffer ops.
  if (llvm::is_contained({OpCode::BUFFER_STORE_DWORDX3,
                          OpCode::BUFFER_STORE_DWORDX4,
                          OpCode::BUFFER_STORE_DWORDX3_IDXEN,
                          OpCode::BUFFER_STORE_DWORDX4_IDXEN},
                         instOp.getOpCode())) {

    // If the dynamic offset is not set, there is no hazard.
    if (!storeOp.getDynamicOff())
      return;
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             *storeOp.getDataOperand().get(),
                             getInstCounts(0)));
    return;
  }

  // Handle global ops.
  if (instOp.hasProp(InstProp::Global)) {
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this),
                             *storeOp.getDataOperand().get(),
                             getInstCounts(0)));
    return;
  }
  // TODO: FLAT_ATOMIC_[F]CMPSWAP_X2, BUFFER_STORE_FORMAT_XYZ/XYZW,
  // BUFFER_ATOMIC_[F]CMPSWAP_X2 not implemented in AMDGCN.
}

bool CDNA3StoreHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  if (!instOp.hasProp(InstProp::IsValu))
    return false;

  auto storeOp = cast<StoreOpInterface>(hazard.getOp());
  AMDGCNRegisterTypeInterface writeRegTy =
      cast<AMDGCNRegisterTypeInterface>(storeOp.getDataOperand().getType());

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
bool CDNA3ValuSgprVmemHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                            ISAVersion isaVer) const {
  return instOp.supportsISA(isaVer) && instOp.hasProp(InstProp::IsValu);
}

void CDNA3ValuSgprVmemHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!instOp.hasProp(InstProp::IsValu))
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

  if (!instOp.hasProp(InstProp::IsVmem))
    return false;

  OpOperand *sgprOperand = hazard.getOperand();
  if (!sgprOperand)
    return false;

  auto sgprRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(sgprOperand->get().getType());
  if (!sgprRegTy || sgprRegTy.getRegisterKind() != RegisterKind::SGPR)
    return false;

  for (Value input : instOp.getInstIns()) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    if (inputRegTy && inputRegTy.getRegisterKind() == RegisterKind::SGPR &&
        checkOverlap(inputRegTy, sgprRegTy))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Case 11: SaluM0GdsSendmsgHazard
//===----------------------------------------------------------------------===//
bool CDNA3SaluM0GdsSendmsgHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3ValuVgprDppHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3ExecDppHazardAttr::matchInst(AMDGCNInstOpInterface,
                                       ISAVersion) const {
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
bool CDNA3MixedVccConstantHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SetRegTrapstsRfeHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SaluM0LdsHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3SaluM0MoverelHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3ValuSgprValuReadHazardAttr::matchInst(AMDGCNInstOpInterface,
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
// VALU writes VGPRn -> v_readfirstlane_b32 reads VGPRn (1 V_NOP)
//===----------------------------------------------------------------------===//
bool CDNA3ValuVgprReadlaneHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                                ISAVersion isaVer) const {
  return instOp.supportsISA(isaVer) && instOp.hasProp(InstProp::IsValu);
}

void CDNA3ValuVgprReadlaneHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!instOp.hasProp(InstProp::IsValu))
    return;

  for (OpOperand &operand : getOpOperands(instOp.getInstOuts())) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(operand.get().getType());
    if (!regTy || regTy.getRegisterKind() != RegisterKind::VGPR)
      continue;

    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this), operand,
                             getInstCounts(0)));
  }
}

bool CDNA3ValuVgprReadlaneHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  if (!isa<VReadfirstlaneB32>(instOp.getOperation()))
    return false;

  OpOperand *vgprOperand = hazard.getOperand();
  if (!vgprOperand)
    return false;

  auto vgprRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(vgprOperand->get().getType());
  if (!vgprRegTy || vgprRegTy.getRegisterKind() != RegisterKind::VGPR)
    return false;

  for (Value input : instOp.getInstIns()) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    if (inputRegTy && inputRegTy.getRegisterKind() == RegisterKind::VGPR &&
        checkOverlap(inputRegTy, vgprRegTy))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Case 20: OpselSdwaHazard
//===----------------------------------------------------------------------===//
bool CDNA3OpselSdwaHazardAttr::matchInst(AMDGCNInstOpInterface,
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
bool CDNA3TransOpHazardAttr::matchInst(AMDGCNInstOpInterface,
                                       ISAVersion) const {
  return false; // TODO: Trans op detection not implemented.
}

void CDNA3TransOpHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3TransOpHazardAttr::isHazardTriggered(const Hazard &,
                                               AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// CDNA3 Section 7.5 - Dependency Resolution: Required Independent Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Case NonDLOpsValuMfmaHazard
//===----------------------------------------------------------------------===//
bool CDNA3NonDLOpsValuMfmaHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                                ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  // Non-DLops VALU = IsValu but NOT MMA (V_MFMA/V_SMFMA).
  return instOp.hasProp(InstProp::IsValu) && !instOp.hasProp(InstProp::Mma);
}

void CDNA3NonDLOpsValuMfmaHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!instOp.hasProp(InstProp::IsValu) || instOp.hasProp(InstProp::Mma))
    return;

  for (OpOperand &operand : getOpOperands(instOp.getInstOuts())) {
    auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(operand.get().getType());
    if (!regTy || !regTy.hasAllocatedSemantics())
      continue;
    // Only VGPR outputs - MFMA reads VGPR (AGPR uses different path).
    if (regTy.getRegisterKind() != RegisterKind::VGPR)
      continue;
    hazards.push_back(Hazard(cast<HazardCheckerAttrInterface>(*this), operand,
                             getInstCounts(0)));
  }
}

bool CDNA3NonDLOpsValuMfmaHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  if (!instOp.hasProp(InstProp::Mma))
    return false;

  OpOperand *vgprOperand = hazard.getOperand();
  if (!vgprOperand)
    return false;

  auto vgprRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(vgprOperand->get().getType());
  if (!vgprRegTy || vgprRegTy.getRegisterKind() != RegisterKind::VGPR)
    return false;

  // V_MFMA*/V_SMFMA* read VGPR as SrcC (or SrcA/B). Check inputs.
  return llvm::any_of(instOp.getInstIns(), [&](Value input) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    return inputRegTy && checkOverlap(inputRegTy, vgprRegTy);
  });
}

//===----------------------------------------------------------------------===//
// Case DLOpsWriteVgprHazard
//===----------------------------------------------------------------------===//
// TODO: DL ops (XDLOP dot product instructions) not implemented in AMDGCN.
bool CDNA3DLOpsWriteVgprHazardAttr::matchInst(AMDGCNInstOpInterface,
                                              ISAVersion) const {
  return false; // TODO: XDLOP instructions not implemented in AMDGCN.
}

void CDNA3DLOpsWriteVgprHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3DLOpsWriteVgprHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case XdlWriteVgprXdlReadSrcCExactHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::matchInst(
    AMDGCNInstOpInterface instOp, ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  return instOp.hasProp(InstProp::Mma);
}

void CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  OpOperand *vdst = getMaiVdst(instOp.getOperation());
  if (!vdst)
    return;

  auto vdstRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(vdst->get().getType());
  assert(vdstRegTy && vdstRegTy.hasAllocatedSemantics() &&
         "vdst must have allocated register semantics");

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  int16_t caseNum = getMfmaPassCase(instOp.getOpCode());
  if (caseNum < 0)
    return; // 8/16-pass MFMA not implemented

  MLIRContext *ctx = instOp.getOperation()->getContext();
  auto attr = CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::get(ctx, caseNum);
  hazards.push_back(Hazard(attr, *vdst, attr.getInstCounts(caseNum)));
}

bool CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  int16_t caseNum = getCaseNum();

  // Check that the second instruction is XDL/V_SMFMA* reading SrcC exactly
  // same.
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return false;

  Value srcC;
  if (auto mai = dyn_cast<inst::VOP3PMAIOp>(instOp.getOperation()))
    srcC = mai.getC();
  else if (auto scaledMai =
               dyn_cast<inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    srcC = scaledMai.getC();
  else
    srcC = {};
  if (!srcC)
    return false;

  OpOperand *raiserVdst = hazard.getOperand();
  assert(raiserVdst && "Raiser operand cannot be nullptr");

  if (srcC.getType() != raiserVdst->get().getType())
    return false;

  // Case-specific check: the first instruction (raiser) must have pass count
  // matching this hazard's case number.
  auto raiserInstOp = dyn_cast<AMDGCNInstOpInterface>(hazard.getOp());
  if (!raiserInstOp)
    return false;

  assert(raiserInstOp.getOpCode() != OpCode::Invalid &&
         "raiser must have a valid opcode");
  int16_t raiserPassCase = getMfmaPassCase(raiserInstOp.getOpCode());
  return raiserPassCase >= 0 && raiserPassCase == caseNum;
}

//===----------------------------------------------------------------------===//
// Case XdlWriteVgprXdlReadSrcCOverlapHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlWriteVgprXdlReadSrcCOverlapHazardAttr::matchInst(
    AMDGCNInstOpInterface, ISAVersion) const {
  return false; // TODO: XDL/V_SMFMA* write VGPR detection.
}

void CDNA3XdlWriteVgprXdlReadSrcCOverlapHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3XdlWriteVgprXdlReadSrcCOverlapHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case XdlWriteVgprSdgemmReadSrcCHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlWriteVgprSdgemmReadSrcCHazardAttr::matchInst(AMDGCNInstOpInterface,
                                                          ISAVersion) const {
  return false; // TODO: XDL/V_SMFMA* write VGPR detection.
}

void CDNA3XdlWriteVgprSdgemmReadSrcCHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3XdlWriteVgprSdgemmReadSrcCHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case XdlWriteVgprMfmaReadSrcABHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::matchInst(
    AMDGCNInstOpInterface instOp, ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  return instOp.hasProp(InstProp::Mma);
}

void CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return;

  OpOperand *vdst = getMaiVdst(instOp.getOperation());
  if (!vdst)
    return;

  auto vdstRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(vdst->get().getType());
  assert(vdstRegTy && vdstRegTy.hasAllocatedSemantics() &&
         "vdst must have allocated register semantics");

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  int16_t caseNum = getMfmaPassCase(instOp.getOpCode());
  if (caseNum < 0)
    return; // 8/16-pass MFMA not implemented

  MLIRContext *ctx = instOp.getOperation()->getContext();
  auto attr = CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::get(ctx, caseNum);
  hazards.push_back(Hazard(attr, *vdst, attr.getInstCounts(caseNum)));
}

bool CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  int16_t caseNum = getCaseNum();

  // Trigger: V_MFMA* or V_SMFMA* reads the raiser's vdst as SrcA or SrcB.
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return false;

  Value srcA, srcB;
  if (auto mai = dyn_cast<inst::VOP3PMAIOp>(instOp.getOperation())) {
    srcA = mai.getA();
    srcB = mai.getB();
  } else if (auto scaledMai =
                 dyn_cast<inst::VOP3PScaledMAIOp>(instOp.getOperation())) {
    srcA = scaledMai.getA();
    srcB = scaledMai.getB();
  } else {
    return false;
  }

  OpOperand *raiserVdst = hazard.getOperand();
  if (!raiserVdst)
    return false;

  auto raiserRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(raiserVdst->get().getType());
  if (!raiserRegTy || !raiserRegTy.hasAllocatedSemantics())
    return false;

  // Check if SrcA or SrcB overlaps with the raiser's vdst.
  auto srcARegTy = dyn_cast<AMDGCNRegisterTypeInterface>(srcA.getType());
  auto srcBRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(srcB.getType());
  bool hasSrcABOverlap = (srcARegTy && checkOverlap(srcARegTy, raiserRegTy)) ||
                         (srcBRegTy && checkOverlap(srcBRegTy, raiserRegTy));
  if (!hasSrcABOverlap)
    return false;

  // Case-specific check: the first instruction (raiser) must have pass count
  // matching this hazard's case number.
  auto raiserInstOp = dyn_cast<AMDGCNInstOpInterface>(hazard.getOp());
  if (!raiserInstOp)
    return false;

  assert(raiserInstOp.getOpCode() != OpCode::Invalid &&
         "raiser must have a valid opcode");
  int16_t raiserPassCase = getMfmaPassCase(raiserInstOp.getOpCode());
  return raiserPassCase >= 0 && raiserPassCase == caseNum;
}

//===----------------------------------------------------------------------===//
// Case CDNA4 XdlWriteVgprMfmaReadSrcABHazard
//===----------------------------------------------------------------------===//
bool CDNA4XdlWriteVgprMfmaReadSrcABHazardAttr::matchInst(
    AMDGCNInstOpInterface instOp, ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  return instOp.hasProp(InstProp::Mma);
}

void CDNA4XdlWriteVgprMfmaReadSrcABHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return;

  OpOperand *vdst = getMaiVdst(instOp.getOperation());
  if (!vdst)
    return;

  auto vdstRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(vdst->get().getType());
  assert(vdstRegTy && vdstRegTy.hasAllocatedSemantics() &&
         "vdst must have allocated register semantics");

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  int16_t caseNum = getMfmaPassCase(instOp.getOpCode());
  if (caseNum < 0)
    return;

  MLIRContext *ctx = instOp.getOperation()->getContext();
  auto attr = CDNA4XdlWriteVgprMfmaReadSrcABHazardAttr::get(ctx, caseNum);
  hazards.push_back(Hazard(attr, *vdst, attr.getInstCounts(caseNum)));
}

bool CDNA4XdlWriteVgprMfmaReadSrcABHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  assert(hazard.getHazard() == *this && "Hazard mismatch");

  int16_t caseNum = getCaseNum();

  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return false;

  Value srcA, srcB;
  if (auto mai = dyn_cast<inst::VOP3PMAIOp>(instOp.getOperation())) {
    srcA = mai.getA();
    srcB = mai.getB();
  } else if (auto scaledMai =
                 dyn_cast<inst::VOP3PScaledMAIOp>(instOp.getOperation())) {
    srcA = scaledMai.getA();
    srcB = scaledMai.getB();
  } else {
    return false;
  }

  OpOperand *raiserVdst = hazard.getOperand();
  if (!raiserVdst)
    return false;

  auto raiserRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(raiserVdst->get().getType());
  if (!raiserRegTy || !raiserRegTy.hasAllocatedSemantics())
    return false;

  auto srcARegTy = dyn_cast<AMDGCNRegisterTypeInterface>(srcA.getType());
  auto srcBRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(srcB.getType());
  bool hasSrcABOverlap = (srcARegTy && checkOverlap(srcARegTy, raiserRegTy)) ||
                         (srcBRegTy && checkOverlap(srcBRegTy, raiserRegTy));
  if (!hasSrcABOverlap)
    return false;

  auto raiserInstOp = dyn_cast<AMDGCNInstOpInterface>(hazard.getOp());
  if (!raiserInstOp)
    return false;

  assert(raiserInstOp.getOpCode() != OpCode::Invalid &&
         "raiser must have a valid opcode");
  int16_t raiserPassCase = getMfmaPassCase(raiserInstOp.getOpCode());
  return raiserPassCase >= 0 && raiserPassCase == caseNum;
}

//===----------------------------------------------------------------------===//
// Case XdlWriteVgprVmemValuHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlWriteVgprVmemValuHazardAttr::matchInst(
    AMDGCNInstOpInterface instOp, ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  return instOp.hasProp(InstProp::Mma);
}

void CDNA3XdlWriteVgprVmemValuHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return;

  OpOperand *vdst = getMaiVdst(instOp.getOperation());
  if (!vdst)
    return;

  auto vdstRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(vdst->get().getType());
  assert(vdstRegTy && vdstRegTy.hasAllocatedSemantics() &&
         "vdst must have allocated register semantics");

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  int16_t caseNum = getMfmaPassCase(instOp.getOpCode());
  if (caseNum < 0)
    return; // 8/16-pass MFMA not implemented

  MLIRContext *ctx = instOp.getOperation()->getContext();
  auto attr = CDNA3XdlWriteVgprVmemValuHazardAttr::get(ctx, caseNum);
  hazards.push_back(Hazard(attr, *vdst, attr.getInstCounts(caseNum)));
}

bool CDNA3XdlWriteVgprVmemValuHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  auto hazardAttr =
      dyn_cast<CDNA3XdlWriteVgprVmemValuHazardAttr>(hazard.getHazard());
  if (!hazardAttr)
    return false;

  // instOp must be VMEM, L/GDS, FLAT, Export, or VALU.
  if (!instOp.hasAnyProps({InstProp::IsValu, InstProp::IsVmem, InstProp::Dsmem,
                           InstProp::Flat, InstProp::Global,
                           InstProp::Buffer}) ||
      instOp.hasProp(InstProp::Mma))
    return false;

  int16_t caseNum = hazardAttr.getCaseNum();

  OpOperand *raiserVdst = hazard.getOperand();
  if (!raiserVdst)
    return false;

  auto raiserRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(raiserVdst->get().getType());
  assert(raiserRegTy && raiserRegTy.hasAllocatedSemantics() &&
         "raiser vdst must have allocated register semantics");

  RegisterKind dstKind = raiserRegTy.getRegisterKind();

  // Check for register conflict: RAW (input overlap) or WAW (output overlap).
  bool hasConflict = false;
  for (Value input : instOp.getInstIns()) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    if (inputRegTy && inputRegTy.getRegisterKind() == dstKind &&
        checkOverlap(inputRegTy, raiserRegTy)) {
      hasConflict = true;
      break;
    }
  }
  if (!hasConflict) {
    for (Value output : instOp.getInstOuts()) {
      auto outputRegTy =
          dyn_cast<AMDGCNRegisterTypeInterface>(output.getType());
      if (outputRegTy && outputRegTy.getRegisterKind() == dstKind &&
          checkOverlap(outputRegTy, raiserRegTy)) {
        hasConflict = true;
        break;
      }
    }
  }

  if (!hasConflict)
    return false;

  // Case-specific check: the first instruction (raiser) must have pass count
  // matching this hazard's case number.
  auto raiserInstOp = dyn_cast<AMDGCNInstOpInterface>(hazard.getOp());
  if (!raiserInstOp)
    return false;

  assert(raiserInstOp.getOpCode() != OpCode::Invalid &&
         "raiser must have a valid opcode");
  int16_t raiserPassCase = getMfmaPassCase(raiserInstOp.getOpCode());
  return raiserPassCase >= 0 && raiserPassCase == caseNum;
}

//===----------------------------------------------------------------------===//
// Case CDNA4 XdlWriteVgprVmemValuHazard
//===----------------------------------------------------------------------===//
bool CDNA4XdlWriteVgprVmemValuHazardAttr::matchInst(
    AMDGCNInstOpInterface instOp, ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  return instOp.hasProp(InstProp::Mma);
}

void CDNA4XdlWriteVgprVmemValuHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!isa<inst::VOP3PMAIOp, inst::VOP3PScaledMAIOp>(instOp.getOperation()))
    return;

  OpOperand *vdst = getMaiVdst(instOp.getOperation());
  if (!vdst)
    return;

  auto vdstRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(vdst->get().getType());
  assert(vdstRegTy && vdstRegTy.hasAllocatedSemantics() &&
         "vdst must have allocated register semantics");

  if (instOp.getOpCode() == OpCode::Invalid)
    return;

  int16_t caseNum = getMfmaPassCase(instOp.getOpCode());
  if (caseNum < 0)
    return;

  MLIRContext *ctx = instOp.getOperation()->getContext();
  auto attr = CDNA4XdlWriteVgprVmemValuHazardAttr::get(ctx, caseNum);
  hazards.push_back(Hazard(attr, *vdst, attr.getInstCounts(caseNum)));
}

bool CDNA4XdlWriteVgprVmemValuHazardAttr::isHazardTriggered(
    const Hazard &hazard, AMDGCNInstOpInterface instOp) const {
  auto hazardAttr =
      dyn_cast<CDNA4XdlWriteVgprVmemValuHazardAttr>(hazard.getHazard());
  if (!hazardAttr)
    return false;

  if (!instOp.hasAnyProps({InstProp::IsValu, InstProp::IsVmem, InstProp::Dsmem,
                           InstProp::Flat, InstProp::Global,
                           InstProp::Buffer}) ||
      instOp.hasProp(InstProp::Mma))
    return false;

  int16_t caseNum = hazardAttr.getCaseNum();

  OpOperand *raiserVdst = hazard.getOperand();
  if (!raiserVdst)
    return false;

  auto raiserRegTy =
      dyn_cast<AMDGCNRegisterTypeInterface>(raiserVdst->get().getType());
  assert(raiserRegTy && raiserRegTy.hasAllocatedSemantics() &&
         "raiser vdst must have allocated register semantics");

  RegisterKind dstKind = raiserRegTy.getRegisterKind();

  bool hasConflict = false;
  for (Value input : instOp.getInstIns()) {
    auto inputRegTy = dyn_cast<AMDGCNRegisterTypeInterface>(input.getType());
    if (inputRegTy && inputRegTy.getRegisterKind() == dstKind &&
        checkOverlap(inputRegTy, raiserRegTy)) {
      hasConflict = true;
      break;
    }
  }
  if (!hasConflict) {
    for (Value output : instOp.getInstOuts()) {
      auto outputRegTy =
          dyn_cast<AMDGCNRegisterTypeInterface>(output.getType());
      if (outputRegTy && outputRegTy.getRegisterKind() == dstKind &&
          checkOverlap(outputRegTy, raiserRegTy)) {
        hasConflict = true;
        break;
      }
    }
  }

  if (!hasConflict)
    return false;

  auto raiserInstOp = dyn_cast<AMDGCNInstOpInterface>(hazard.getOp());
  if (!raiserInstOp)
    return false;

  assert(raiserInstOp.getOpCode() != OpCode::Invalid &&
         "raiser must have a valid opcode");
  int16_t raiserPassCase = getMfmaPassCase(raiserInstOp.getOpCode());
  return raiserPassCase >= 0 && raiserPassCase == caseNum;
}

//===----------------------------------------------------------------------===//
// Case SgemmWriteVgprXdlReadSrcCExactHazard
//===----------------------------------------------------------------------===//
bool CDNA3SgemmWriteVgprXdlReadSrcCExactHazardAttr::matchInst(
    AMDGCNInstOpInterface, ISAVersion) const {
  return false; // TODO: SGEMM write VGPR detection.
}

void CDNA3SgemmWriteVgprXdlReadSrcCExactHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SgemmWriteVgprXdlReadSrcCExactHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case SgemmWriteVgprXdlReadSrcCOverlapHazard
//===----------------------------------------------------------------------===//
bool CDNA3SgemmWriteVgprXdlReadSrcCOverlapHazardAttr::matchInst(
    AMDGCNInstOpInterface, ISAVersion) const {
  return false; // TODO: SGEMM write VGPR detection.
}

void CDNA3SgemmWriteVgprXdlReadSrcCOverlapHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SgemmWriteVgprXdlReadSrcCOverlapHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case SgemmWriteVgprSdgemmReadSrcCHazard
//===----------------------------------------------------------------------===//
bool CDNA3SgemmWriteVgprSdgemmReadSrcCHazardAttr::matchInst(
    AMDGCNInstOpInterface, ISAVersion) const {
  return false; // TODO: SGEMM write VGPR detection.
}

void CDNA3SgemmWriteVgprSdgemmReadSrcCHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SgemmWriteVgprSdgemmReadSrcCHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case SgemmWriteVgprMfmaReadSrcABHazard
//===----------------------------------------------------------------------===//
bool CDNA3SgemmWriteVgprMfmaReadSrcABHazardAttr::matchInst(
    AMDGCNInstOpInterface, ISAVersion) const {
  return false; // TODO: SGEMM write VGPR detection.
}

void CDNA3SgemmWriteVgprMfmaReadSrcABHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SgemmWriteVgprMfmaReadSrcABHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case SgemmWriteVgprVmemValuHazard
//===----------------------------------------------------------------------===//
bool CDNA3SgemmWriteVgprVmemValuHazardAttr::matchInst(AMDGCNInstOpInterface,
                                                      ISAVersion) const {
  return false; // TODO: SGEMM write VGPR detection.
}

void CDNA3SgemmWriteVgprVmemValuHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3SgemmWriteVgprVmemValuHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case Mfma16x16x4F64WriteVgprHazard
//===----------------------------------------------------------------------===//
bool CDNA3Mfma16x16x4F64WriteVgprHazardAttr::matchInst(AMDGCNInstOpInterface,
                                                       ISAVersion) const {
  return false; // TODO: V_MFMA_16x16x4_F64 detection.
}

void CDNA3Mfma16x16x4F64WriteVgprHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3Mfma16x16x4F64WriteVgprHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case Mfma4x4x4F64WriteVgprHazard
//===----------------------------------------------------------------------===//
bool CDNA3Mfma4x4x4F64WriteVgprHazardAttr::matchInst(AMDGCNInstOpInterface,
                                                     ISAVersion) const {
  return false; // TODO: V_MFMA_4x4x4_F64 detection.
}

void CDNA3Mfma4x4x4F64WriteVgprHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3Mfma4x4x4F64WriteVgprHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Case VcmpxExecMfmaHazard
//===----------------------------------------------------------------------===//
bool CDNA3VcmpxExecMfmaHazardAttr::matchInst(AMDGCNInstOpInterface instOp,
                                             ISAVersion isaVer) const {
  if (!instOp.supportsISA(isaVer))
    return false;
  // V_CMPX* instructions write EXEC. TODO: Check for V_CMPX* opcode.
  return instOp.hasProp(InstProp::IsValu);
}

void CDNA3VcmpxExecMfmaHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface instOp, SmallVectorImpl<Hazard> &hazards) const {
  if (!instOp.hasProp(InstProp::IsValu))
    return;
  // TODO: Check if instruction is V_CMPX* that writes EXEC.
  (void)instOp;
  (void)hazards;
}

bool CDNA3VcmpxExecMfmaHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface instOp) const {
  // TODO: Check if instruction is V_MFMA* (Mma covers matrix ops).
  return instOp.hasProp(InstProp::Mma);
}

//===----------------------------------------------------------------------===//
// Case XdlSmfmaReadSrcCValuWriteHazard
//===----------------------------------------------------------------------===//
bool CDNA3XdlSmfmaReadSrcCValuWriteHazardAttr::matchInst(AMDGCNInstOpInterface,
                                                         ISAVersion) const {
  return false; // TODO: XDL/SMFMA read VGPR SrcC detection.
}

void CDNA3XdlSmfmaReadSrcCValuWriteHazardAttr::populateHazardsFor(
    AMDGCNInstOpInterface, SmallVectorImpl<Hazard> &) const {}

bool CDNA3XdlSmfmaReadSrcCValuWriteHazardAttr::isHazardTriggered(
    const Hazard &, AMDGCNInstOpInterface) const {
  return false;
}

//===----------------------------------------------------------------------===//
// Hazard manager implementation
//===----------------------------------------------------------------------===//

void HazardManager::getHazardRaisersFor(
    ISAVersion version,
    SmallVectorImpl<HazardRaiserAttrInterface> &hazardRaisers) {
  hazardRaisers.clear();
  if (version == ISAVersion::CDNA3 || version == ISAVersion::CDNA4) {
    MLIRContext *ctx = getTopOp()->getContext();
    // Hazards with identical NOP counts on CDNA3 and CDNA4.
    hazardRaisers.push_back(CDNA3StoreHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3StoreWriteDataHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3VccExecVcczExeczHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3ValuSgprVmemHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3ValuVgprReadlaneHazardAttr::get(ctx));
    hazardRaisers.push_back(CDNA3NonDLOpsValuMfmaHazardAttr::get(ctx));
    hazardRaisers.push_back(
        CDNA3XdlWriteVgprXdlReadSrcCExactHazardAttr::get(ctx));
    // XDL write -> SrcA/B and XDL write -> VMEM/VALU: CDNA4 needs +1 NOP for
    // 4-pass and higher.
    if (version == ISAVersion::CDNA4) {
      hazardRaisers.push_back(
          CDNA4XdlWriteVgprMfmaReadSrcABHazardAttr::get(ctx));
      hazardRaisers.push_back(CDNA4XdlWriteVgprVmemValuHazardAttr::get(ctx));
    } else {
      assert(version == ISAVersion::CDNA3 && "unexpected ISA version");
      hazardRaisers.push_back(
          CDNA3XdlWriteVgprMfmaReadSrcABHazardAttr::get(ctx));
      hazardRaisers.push_back(CDNA3XdlWriteVgprVmemValuHazardAttr::get(ctx));
    }
  }
}

void HazardManager::populateHazardsFor(
    ISAVersion version,
    ArrayRef<HazardRaiserAttrInterface> additionalHazardRaisers) {
  hazardAttrs.clear();
  opcodeToHazardAttrs.clear();

  // Collect all hazard raisers for the given ISA version.
  SmallVector<HazardRaiserAttrInterface> hazardRaisers;
  getHazardRaisersFor(version, hazardRaisers);
  llvm::append_range(hazardRaisers, additionalHazardRaisers);

  // Since we are matching by opcode, we don't need to visit the same opcode
  // twice.
  llvm::SmallDenseSet<OpCode> seen;

  // Walk the top operation and populate the hazard raisers by opcode.
  getTopOp()->walk([&](AMDGCNInstOpInterface instOp) {
    OpCode opcode = instOp.getOpCode();

    // Skip instructions that don't have an opcode or have already been visited.
    if (opcode == OpCode::Invalid || !seen.insert(opcode).second)
      return;

    // Check if the instruction has a hazard raiser.
    for (HazardRaiserAttrInterface hazardRaiser : hazardRaisers) {
      if (!hazardRaiser.matchInst(instOp, version))
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
  OpCode opcode = instOp.getOpCode();
  if (opcode == OpCode::Invalid) {
    LDBG() << "- no opcode for instruction";
    return success();
  }

  // Lookup if there are any hazard raisers for the given instruction.
  auto [start, end] = opcodeToHazardAttrs.lookup_or(opcode, {-1, -1});
  if (start == -1 || end == -1) {
    LDBG() << "- no hazard patterns for instruction";
    return success();
  }

  // Check if all output and input operands have allocated semantics, bail if
  // not.
  auto isValidOperand = [](Value v) {
    if (Operation *op = v.getDefiningOp(); op && m_Constant().match(op))
      return true; // Constants are always valid.
    auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
    return regTy && regTy.hasAllocatedSemantics();
  };

  if (!llvm::all_of(instOp.getInstOuts(), isValidOperand))
    return instOp.emitError("output operands must have allocated semantics");
  if (!llvm::all_of(instOp.getInstIns(), isValidOperand))
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
