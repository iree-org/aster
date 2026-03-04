//===- Hazards.h - AMDGCN hazard detection -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN hazard detection utilities and hazard managers.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_HAZARDS_H
#define ASTER_DIALECT_AMDGCN_IR_HAZARDS_H

#include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.h"
#include "llvm/ADT/PointerUnion.h"
#include <cstdint>

namespace mlir {
class DominanceInfo;
namespace aster::amdgcn {
/// Object to model instruction counts required to resolve a hazard.
struct InstCounts {
  InstCounts(int8_t numVector = 0, int8_t numScalar = 0,
             int8_t numDataShare = 0)
      : numVector(numVector), numScalar(numScalar), numDataShare(numDataShare) {
    assert(numVector >= 0 && "numVector cannot be negative");
    assert(numScalar >= 0 && "numScalar cannot be negative");
    assert(numDataShare >= 0 && "numDataShare cannot be negative");
  }

  bool operator==(const InstCounts &other) const {
    return numVector == other.numVector && numScalar == other.numScalar &&
           numDataShare == other.numDataShare;
  }

  bool operator<(const InstCounts &other) const {
    return std::make_tuple(numVector, numScalar, numDataShare) <
           std::make_tuple(other.numVector, other.numScalar,
                           other.numDataShare);
  }

  /// Check if there are no instructions required.
  bool isInactive() const {
    return numVector <= 0 && numScalar <= 0 && numDataShare <= 0;
  }

  /// Get the number of vector instructions required.
  int8_t getNumVector() const { return numVector; }

  /// Get the number of scalar instructions required.
  int8_t getNumScalar() const { return numScalar; }

  /// Get the number of data share instructions required.
  int8_t getNumDataShare() const { return numDataShare; }

  /// Clear the instruction counts.
  void clear() {
    numVector = 0;
    numScalar = 0;
    numDataShare = 0;
  }

  /// Join the instruction counts with the minimum of the two.
  void joinWithMin(const InstCounts &other) {
    numVector = std::min(numVector, other.numVector);
    numScalar = std::min(numScalar, other.numScalar);
    numDataShare = std::min(numDataShare, other.numDataShare);
  }

  /// Join the instruction counts with the maximum of the two.
  void joinWithMax(const InstCounts &other) {
    numVector = std::max(numVector, other.numVector);
    numScalar = std::max(numScalar, other.numScalar);
    numDataShare = std::max(numDataShare, other.numDataShare);
  }

  /// Decrement the instruction counts by the given amounts. This method always
  /// clamps the counts to [0, numInst].
  void decrementCount(int8_t vInst, int8_t sInst, int8_t dsInst) {
    numVector = std::min(std::max<int8_t>(0, numVector - vInst), numVector);
    numScalar = std::min(std::max<int8_t>(0, numScalar - sInst), numScalar);
    numDataShare =
        std::min(std::max<int8_t>(0, numDataShare - dsInst), numDataShare);
  }

  void decrementCount(const InstCounts &other) {
    decrementCount(other.numVector, other.numScalar, other.numDataShare);
  }

  /// Print the instruction counts.
  void print(llvm::raw_ostream &os) const {
    os << "{v:" << (int)numVector << ", s:" << (int)numScalar
       << ", ds:" << (int)numDataShare << "}";
  }

private:
  /// The number of vector instructions required.
  int8_t numVector = 0;
  /// The number of scalar instructions required.
  int8_t numScalar = 0;
  /// The number of data share instructions required.
  int8_t numDataShare = 0;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const InstCounts &counts) {
  counts.print(os);
  return os;
}

/// A hazard raised by an operation or operand. This class is used to track the
/// state of a hazard and handle it when it is triggered.
struct Hazard {
  Hazard() = default;
  Hazard(HazardAttrInterface hazard, Operation *op,
         const InstCounts &instCounts)
      : hazard(hazard), opOrOperand(op), instCounts(instCounts) {
    assert(op != nullptr && "op cannot be nullptr");
  }
  Hazard(HazardAttrInterface hazard, OpOperand &operand,
         const InstCounts &instCounts)
      : hazard(hazard), opOrOperand(&operand), instCounts(instCounts) {
    assert(operand.getOwner() != nullptr && "operand cannot be nullptr");
  }

  bool operator==(const Hazard &other) const {
    return hazard == other.hazard && opOrOperand == other.opOrOperand;
  }

  bool operator!=(const Hazard &other) const { return !(*this == other); }

  bool operator<(const Hazard &other) const {
    return std::make_tuple(hazard.getAsOpaquePointer(), getOp(),
                           opOrOperand.getOpaqueValue()) <
           std::make_tuple(other.hazard.getAsOpaquePointer(), other.getOp(),
                           other.opOrOperand.getOpaqueValue());
  }

  /// Compare less two hazards by lower inst count, dominance of the operation,
  /// operand number, and hazard. This method should be used for deterministic
  /// sorting.
  bool compare(const Hazard &other, DominanceInfo &domInfo) const;

  /// Check if the hazard is valid.
  bool isValid() const { return hazard != nullptr && !opOrOperand.isNull(); }

  /// Check if the hazard is active.
  bool isActive() const { return !instCounts.isInactive(); }

  /// Clear the hazard.
  void clear() {
    hazard = nullptr;
    opOrOperand = nullptr;
    instCounts.clear();
  }

  /// Clear the hazard if it is inactive.
  void clearIfInactive() {
    if (!isActive())
      clear();
  }

  /// Get the hazard attribute.
  HazardAttrInterface getHazard() const { return hazard; }

  /// Get the operation that raises the hazard.
  Operation *getOp() const {
    return isa<Operation *>(opOrOperand)
               ? dyn_cast<Operation *>(opOrOperand)
               : dyn_cast<OpOperand *>(opOrOperand)->getOwner();
  }

  /// Get the operand that raises the hazard.
  OpOperand *getOperand() const { return dyn_cast<OpOperand *>(opOrOperand); }

  /// Get the instruction counts required to resolve the hazard.
  InstCounts getInstCounts() const { return instCounts; }
  InstCounts &getInstCounts() { return instCounts; }

  /// Join two hazards. If the hazards are conflicting, return a hazard with
  /// the elementwise minimum of the instruction counts, otherwise return an
  /// inactive hazard.
  Hazard join(const Hazard &other) const {
    if (other.hazard != hazard || other.opOrOperand != opOrOperand)
      return Hazard();
    Hazard result = *this;
    result.instCounts.joinWithMin(other.instCounts);
    return result;
  }

  /// Print the hazard.
  void print(llvm::raw_ostream &os) const {
    if (!isActive()) {
      os << "<inactive>";
      return;
    }
    os << "{" << hazard << ", "
       << OpWithFlags(getOp(), OpPrintingFlags().skipRegions());
    if (OpOperand *operand = getOperand())
      os << ", " << operand->getOperandNumber();
    else
      os << ", none";
    os << ", " << instCounts << "}";
  }

private:
  /// The raised hazard.
  HazardAttrInterface hazard;
  /// The operation or operand that raises the hazard.
  llvm::PointerUnion<Operation *, OpOperand *> opOrOperand;
  /// The instruction counts required to resolve the hazard.
  InstCounts instCounts;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Hazard &h) {
  h.print(os);
  return os;
}

/// A hazard manager is responsible for computing the hazards for a given
/// operation.
struct HazardManager {
  /// Create a new hazard manager for the given top operation.
  HazardManager(Operation *topOp) : topOp(topOp) {
    assert(topOp != nullptr && "topOp cannot be nullptr");
  }

  virtual ~HazardManager() = default;

  /// Initialize the hazard manager.
  virtual void initialize() {}

  /// Get the hazards for the given operation. Returns failure if the hazards
  /// cannot be computed.
  virtual LogicalResult getHazards(Operation *op,
                                   SmallVectorImpl<Hazard> &hazards) = 0;

  /// Get the top operation of the hazard manager.
  Operation *getTopOp() const { return topOp; }

private:
  Operation *topOp;
};

/// Hazard manager for CDNA3 (GFX940/GFX942) architectures. Computes
/// instruction hazards specific to the CDNA3 ISA.
struct CDNA3Hazards : HazardManager {
  CDNA3Hazards(Operation *topOp) : HazardManager(topOp) { initialize(); }

  void initialize() override;

  /// Get the hazards for the given operation.
  LogicalResult getHazards(Operation *op,
                           SmallVectorImpl<Hazard> &hazards) override {
    if (auto amdgcnInstOp = dyn_cast<AMDGCNInstOpInterface>(op))
      return getHazards(amdgcnInstOp, hazards);
    return success();
  }
  LogicalResult getHazards(AMDGCNInstOpInterface instOp,
                           SmallVectorImpl<Hazard> &hazards);

private:
  /// Cache the hazard attributes.
  SmallVector<HazardAttrInterface> hazardAttrs;
};
} // namespace aster::amdgcn
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_IR_HAZARDS_H
