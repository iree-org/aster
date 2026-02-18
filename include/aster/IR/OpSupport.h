//===- OpSupport.h - ASTER Op support ---------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_OPSUPPORT_H
#define ASTER_IR_OPSUPPORT_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::aster {
/// Get an operand range from a mutable array of operands.
inline OperandRange
getAsOperandRange(llvm::MutableArrayRef<OpOperand> operands) {
  return OperandRange(operands.data(), operands.size());
}

/// Helper function to get the operand segment sizes for an operation. If the
/// operation has the AttrSizedOperandSegments trait, return the operand segment
/// sizes from the properties. Otherwise, return an empty array.
template <typename OpTy>
MutableArrayRef<int32_t> getOperandSegmentSizes(OpTy op) {
  if constexpr (OpTy::template hasTrait<OpTrait::AttrSizedOperandSegments>()) {
    return op.getProperties().operandSegmentSizes;
  }
  return {};
}
template <typename OpTy>
MutableArrayRef<int32_t> getOperandSegmentSizes(OpaqueProperties properties) {
  if constexpr (OpTy::template hasTrait<OpTrait::AttrSizedOperandSegments>()) {
    return properties.as<typename OpTy::Properties *>()->operandSegmentSizes;
  }
  return {};
}

/// Helper function to get the result segment sizes for an operation. If the
/// operation has the AttrSizedResultSegments trait, return the result segment
/// sizes from the properties. Otherwise, return an empty array.
template <typename OpTy>
MutableArrayRef<int32_t> getResultSegmentSizes(OpTy op) {
  if constexpr (OpTy::template hasTrait<OpTrait::AttrSizedResultSegments>()) {
    return op.getProperties().resultSegmentSizes;
  }
  return {};
}
template <typename OpTy>
MutableArrayRef<int32_t> getResultSegmentSizes(OpaqueProperties properties) {
  if constexpr (OpTy::template hasTrait<OpTrait::AttrSizedResultSegments>()) {
    return properties.as<typename OpTy::Properties *>()->resultSegmentSizes;
  }
  return {};
}
} // namespace mlir::aster

namespace llvm {
/// DenseMapInfo for ValueRange, this implementation is adapted from the
/// ArrayRef implementation.
template <>
struct DenseMapInfo<mlir::ValueRange> {
  using PtrType = llvm::PointerUnion<const mlir::Value *, mlir::OpOperand *,
                                     mlir::detail::OpResultImpl *>;
  static inline mlir::ValueRange getEmptyKey() {
    return mlir::ValueRange(DenseMapInfo<PtrType>::getEmptyKey(), 0);
  }

  static inline mlir::ValueRange getTombstoneKey() {
    return mlir::ValueRange(DenseMapInfo<PtrType>::getTombstoneKey(), 0);
  }

  static unsigned getHashValue(const mlir::ValueRange &range) {
    assert(range.getBase() != getEmptyKey().getBase() &&
           "Cannot hash the empty key!");
    assert(range.getBase() != getTombstoneKey().getBase() &&
           "Cannot hash the tombstone key!");
    return (unsigned)(hash_value(range));
  }

  static bool isEqual(const mlir::ValueRange &lhs,
                      const mlir::ValueRange &rhs) {
    if (rhs.getBase() == getEmptyKey().getBase())
      return lhs.getBase() == getEmptyKey().getBase();
    if (rhs.getBase() == getTombstoneKey().getBase())
      return lhs.getBase() == getTombstoneKey().getBase();
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // ASTER_IR_OPSUPPORT_H
