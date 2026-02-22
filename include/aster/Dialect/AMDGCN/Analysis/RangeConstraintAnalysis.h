//===- RangeConstraintAnalysis.h - Range constraint analysis ----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_ANALYSIS_RANGECONSTRAINTANALYSIS_H
#define ASTER_DIALECT_AMDGCN_ANALYSIS_RANGECONSTRAINTANALYSIS_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir::aster {
class SSAMap;
}
namespace mlir::aster::amdgcn {
class AllocaOp;
class MakeRegisterRangeOp;
/// Represents a range constraint.
struct RangeConstraint {
  SmallVector<Value> allocations;
  int32_t alignment = 1;

  /// Print the range constraint.
  void print(llvm::raw_ostream &os) const {
    os << "range_constraint<alignment = " << alignment << ", allocations = [";
    llvm::interleaveComma(allocations, os, [&](Value v) {
      v.printAsOperand(os, OpPrintingFlags());
    });
    os << "]>";
  }

  /// Print the range constraint with SSA value IDs from the given map.
  void print(llvm::raw_ostream &os, const mlir::aster::SSAMap &ssaMap) const;
};

//===----------------------------------------------------------------------===//
// RangeConstraintAnalysis
//===----------------------------------------------------------------------===//

/// This class represents a register range constraint analysis.
struct RangeConstraintAnalysis {
  /// Create a RangeConstraintAnalysis instance from a top-level operation. If
  /// the constraints are not satisfiable, returns failure.
  static FailureOr<RangeConstraintAnalysis> create(Operation *topOp);

  /// Get the range constraints.
  ArrayRef<RangeConstraint> getRanges() const { return constraints; }

  /// Lookup the range constraint for a given value. Returns nullptr if the
  /// value is not in any range.
  const RangeConstraint *lookup(Value value) const {
    auto it = valueToConstraintIdx.find(value);
    if (it == valueToConstraintIdx.end())
      return nullptr;
    return &constraints[it->second];
  }

  /// Get the number of range constraints.
  int64_t size() const { return constraints.size(); }

  /// Get the range constraint for a given index. Returns nullptr if the index
  /// is out of bounds.
  const RangeConstraint *getConstraintOrNull(int64_t idx) const {
    if (idx < 0 || idx >= static_cast<int64_t>(constraints.size()))
      return nullptr;
    return &constraints[idx];
  }

private:
  SmallVector<RangeConstraint> constraints;
  DenseMap<Value, int64_t> valueToConstraintIdx;
};

} // end namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_ANALYSIS_RANGECONSTRAINTANALYSIS_H
