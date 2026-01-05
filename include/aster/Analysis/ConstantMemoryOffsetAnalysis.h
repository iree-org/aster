//===- ConstantMemoryOffsetAnalysis.h - Constant offset analysis ---*-
// C++-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a forward dataflow analysis that propagates constant
// values through affine.apply, index.cast, and amdgcn.vop1.vop1 <v_mov_b32_e32>
// operations. For memory operations, it captures whether the address comes from
// an affine.apply with a constant, separating the constant from the AffineMap
// and operands.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_ANALYSIS_CONSTANTMEMORYOFFSETANALYSIS_H
#define ASTER_ANALYSIS_CONSTANTMEMORYOFFSETANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::aster {

//===----------------------------------------------------------------------===//
// BaseValueInfo - Base class for different types of base values
//===----------------------------------------------------------------------===//

/// Base class for representing different types of base values in memory
/// address calculations.
struct BaseValueInfo {
  enum class Kind { AffineApply, SSAValue };
  Kind kind;

  BaseValueInfo(Kind k) : kind(k) {}
  virtual ~BaseValueInfo() = default;
  virtual bool operator==(const BaseValueInfo &other) const = 0;
  virtual void print(raw_ostream &os) const = 0;
  virtual std::unique_ptr<BaseValueInfo> clone() const = 0;
};

/// Represents a base value that is an SSA value.
struct SSAValueInfo : BaseValueInfo {
  Value ssaValue;

  SSAValueInfo() : BaseValueInfo(Kind::SSAValue), ssaValue(Value()) {}
  SSAValueInfo(Value value) : BaseValueInfo(Kind::SSAValue), ssaValue(value) {}

  static bool classof(const BaseValueInfo *info) {
    return info->kind == Kind::SSAValue;
  }

  bool operator==(const BaseValueInfo &other) const override {
    if (auto *otherSSA = dyn_cast<SSAValueInfo>(&other)) {
      return ssaValue == otherSSA->ssaValue;
    }
    return false;
  }

  void print(raw_ostream &os) const override {
    os << "ssa_value=%" << ssaValue.getAsOpaquePointer();
  }

  std::unique_ptr<BaseValueInfo> clone() const override {
    return std::make_unique<SSAValueInfo>(ssaValue);
  }
};

/// Represents a base value that comes from an affine.apply operation.
struct AffineApplyValueInfo : BaseValueInfo {
  AffineMap affineMap;
  SmallVector<Value> affineOperands;

  AffineApplyValueInfo()
      : BaseValueInfo(Kind::AffineApply), affineMap(AffineMap()) {}
  AffineApplyValueInfo(AffineMap map, ArrayRef<Value> operands)
      : BaseValueInfo(Kind::AffineApply), affineMap(map),
        affineOperands(operands.begin(), operands.end()) {}

  static bool classof(const BaseValueInfo *info) {
    return info->kind == Kind::AffineApply;
  }

  bool operator==(const BaseValueInfo &other) const override {
    if (auto *otherAffine = dyn_cast<AffineApplyValueInfo>(&other)) {
      return affineMap == otherAffine->affineMap &&
             affineOperands == otherAffine->affineOperands;
    }
    return false;
  }

  void print(raw_ostream &os) const override {
    os << "affine_map=";
    affineMap.print(os);
    os << " operands=[";
    for (Value op : affineOperands) {
      os << "%" << op.getAsOpaquePointer() << " ";
    }
    os << "]";
  }

  std::unique_ptr<BaseValueInfo> clone() const override {
    return std::make_unique<AffineApplyValueInfo>(affineMap, affineOperands);
  }

  bool hasAffineMap() const { return affineMap.getNumResults() > 0; }
};

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetInfo - Represents constant offset information
//===----------------------------------------------------------------------===//

/// Represents information about a value that may have a constant component
/// propagated through affine.apply, index.cast, or v_mov_b32_e32 operations.
/// Separates the constant offset from the base value structure.
struct ConstantMemoryOffsetInfo {
  /// The constant offset (defaults to 0)
  int64_t constantOffset = 0;

  /// The base value information (e.g., AffineApplyValueInfo)
  std::unique_ptr<BaseValueInfo> baseValue;

  /// Whether this value has base value information
  bool hasBaseValue() const { return baseValue != nullptr; }

  /// Get the base value as AffineApplyValueInfo if it is one
  const AffineApplyValueInfo *getAffineApplyValueInfo() const {
    if (!baseValue || baseValue->kind != BaseValueInfo::Kind::AffineApply)
      return nullptr;
    return static_cast<const AffineApplyValueInfo *>(baseValue.get());
  }

  /// Whether this value comes from an affine.apply
  bool hasAffineMap() const {
    if (auto *affineInfo = getAffineApplyValueInfo())
      return affineInfo->hasAffineMap();
    return false;
  }

  ConstantMemoryOffsetInfo() = default;
  ConstantMemoryOffsetInfo(int64_t offset) : constantOffset(offset) {}
  ConstantMemoryOffsetInfo(std::unique_ptr<BaseValueInfo> base)
      : constantOffset(0), baseValue(std::move(base)) {}
  ConstantMemoryOffsetInfo(int64_t offset, std::unique_ptr<BaseValueInfo> base)
      : constantOffset(offset), baseValue(std::move(base)) {}

  // Convenience constructors for AffineApplyValueInfo
  ConstantMemoryOffsetInfo(AffineMap map, ArrayRef<Value> operands)
      : baseValue(std::make_unique<AffineApplyValueInfo>(map, operands)) {}
  ConstantMemoryOffsetInfo(int64_t offset, AffineMap map,
                           ArrayRef<Value> operands)
      : constantOffset(offset),
        baseValue(std::make_unique<AffineApplyValueInfo>(map, operands)) {}

  // Convenience constructors for SSAValueInfo
  ConstantMemoryOffsetInfo(Value ssaValue)
      : baseValue(std::make_unique<SSAValueInfo>(ssaValue)) {}
  ConstantMemoryOffsetInfo(int64_t offset, Value ssaValue)
      : constantOffset(offset),
        baseValue(std::make_unique<SSAValueInfo>(ssaValue)) {}

  ConstantMemoryOffsetInfo(const ConstantMemoryOffsetInfo &other)
      : constantOffset(other.constantOffset),
        baseValue(other.baseValue ? other.baseValue->clone() : nullptr) {}

  ConstantMemoryOffsetInfo &operator=(const ConstantMemoryOffsetInfo &other) {
    if (this != &other) {
      constantOffset = other.constantOffset;
      baseValue = other.baseValue ? other.baseValue->clone() : nullptr;
    }
    return *this;
  }

  bool operator==(const ConstantMemoryOffsetInfo &other) const {
    if (constantOffset != other.constantOffset)
      return false;
    if (baseValue == nullptr && other.baseValue == nullptr)
      return true;
    if (baseValue == nullptr || other.baseValue == nullptr)
      return false;
    return *baseValue == *other.baseValue;
  }
};

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetLattice - Lattice for constant offset analysis
//===----------------------------------------------------------------------===//

/// This lattice represents constant offset information at a program point.
/// It tracks constant values propagated through supported operations, keyed by
/// operation.
struct ConstantMemoryOffsetLattice : dataflow::AbstractDenseLattice {
  ConstantMemoryOffsetLattice(LatticeAnchor anchor)
      : AbstractDenseLattice(anchor), isTopState(false) {}

  /// Whether the state is the top state.
  bool isTop() const { return isTopState; }

  /// Whether the state is empty (no constant information).
  bool isEmpty() const { return !isTopState && constantOffsetValues.empty(); }

  /// Set the lattice to top.
  ChangeResult setToTop() {
    if (!isTop()) {
      isTopState = true;
      constantOffsetValues.clear();
      return ChangeResult::Change;
    }
    return ChangeResult::NoChange;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override;

  /// Meet operation for the lattice.
  ChangeResult meet(const ConstantMemoryOffsetLattice &lattice) {
    assert(false && "meet not supported atm");
    return ChangeResult::NoChange;
  }
  ChangeResult meet(const AbstractDenseLattice &lattice) final {
    return meet(static_cast<const ConstantMemoryOffsetLattice &>(lattice));
  }

  /// Join operation for the lattice.
  ChangeResult join(const ConstantMemoryOffsetLattice &lattice) {
    assert(false && "join not supported atm");
    return ChangeResult::NoChange;
  }
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    return join(static_cast<const ConstantMemoryOffsetLattice &>(lattice));
  }

  /// Set the constant information for a value.
  ChangeResult setInfo(Value value, const ConstantMemoryOffsetInfo &newInfo) {
    if (isTopState)
      return ChangeResult::NoChange;
    auto it = constantOffsetValues.find(value);
    if (it == constantOffsetValues.end()) {
      constantOffsetValues.insert({value, newInfo});
      return ChangeResult::Change;
    }
    if (it->second == newInfo)
      return ChangeResult::NoChange;
    it->second = newInfo;
    return ChangeResult::Change;
  }

  /// Get the constant information for a value, or default if not found.
  ConstantMemoryOffsetInfo getInfo(Value value) const {
    auto it = constantOffsetValues.find(value);
    if (it != constantOffsetValues.end())
      return it->second;
    // Return default (offset 0, no base value) if not found
    return ConstantMemoryOffsetInfo(0, value);
  }

  /// Get all value info mappings.
  const llvm::DenseMap<Value, ConstantMemoryOffsetInfo> &getValueInfo() const {
    return constantOffsetValues;
  }

private:
  bool isTopState;
  llvm::DenseMap<Value, ConstantMemoryOffsetInfo> constantOffsetValues;
};

//===----------------------------------------------------------------------===//
// ConstantMemoryOffsetAnalysis
//===----------------------------------------------------------------------===//

/// An analysis that, by going forward along the dataflow graph, propagates
/// constant values through affine.apply, index.cast, and v_mov_b32_e32
/// operations.
class ConstantMemoryOffsetAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<
          ConstantMemoryOffsetLattice> {
public:
  ConstantMemoryOffsetAnalysis(DataFlowSolver &solver)
      : dataflow::DenseForwardDataFlowAnalysis<ConstantMemoryOffsetLattice>(
            solver) {}

  /// Visit an operation and update the lattice state.
  LogicalResult visitOperation(Operation *op,
                               const ConstantMemoryOffsetLattice &before,
                               ConstantMemoryOffsetLattice *after) override;

  /// Visit a block transfer and update the lattice state.
  void visitBlockTransfer(Block *block, ProgramPoint *point, Block *predecessor,
                          const ConstantMemoryOffsetLattice &before,
                          ConstantMemoryOffsetLattice *after) override;

  /// Visit a call control flow transfer and update the lattice state.
  void
  visitCallControlFlowTransfer(CallOpInterface call,
                               dataflow::CallControlFlowAction action,
                               const ConstantMemoryOffsetLattice &before,
                               ConstantMemoryOffsetLattice *after) override;

  /// Visit a region branch control flow transfer and update the lattice state.
  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo,
      const ConstantMemoryOffsetLattice &before,
      ConstantMemoryOffsetLattice *after) override;

  /// Set the lattice to the entry state.
  void setToEntryState(ConstantMemoryOffsetLattice *lattice) override;

private:
  /// Handle propagation when either of the states are top. Returns true if
  /// either state is top.
  bool handleTopPropagation(const ConstantMemoryOffsetLattice &before,
                            ConstantMemoryOffsetLattice *after);
};

} // end namespace mlir::aster

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::aster::ConstantMemoryOffsetLattice)

#endif // ASTER_ANALYSIS_CONSTANTMEMORYOFFSETANALYSIS_H
