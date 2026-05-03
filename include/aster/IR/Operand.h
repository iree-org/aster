//===- Operand.h - Aster operand wrapper ------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Operand wrapper type.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_OPERAND_H
#define ASTER_IR_OPERAND_H

#include "aster/IR/OpSupport.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster {
/// A nullable wrapper around an OpOperand pointer.
struct Operand {
  Operand() = default;
  Operand(OpOperand &operand) : operand(&operand) {}

  explicit operator bool() const { return operand != nullptr; }
  OpOperand *operator->() const { return operand; }

  /// Get the underlying OpOperand pointer.
  OpOperand *get() const { return operand; }

  /// Get the value of the operand, or a null Value if the operand is not set.
  Value getValue() const { return operand ? operand->get() : Value(); }

  /// Get the type of the operand, or a null Type if the operand is not set.
  Type getType() const {
    if (Value val = getValue())
      return val.getType();
    return Type();
  }

private:
  OpOperand *operand = nullptr;
};

/// A mutable ODS operand.
struct MutableOperand {
  MutableOperand() = default;
  MutableOperand(Operation *op, int32_t start, int32_t size, int32_t *segment)
      : op(op), start(start), length(size), segment(segment) {
    assert(op != nullptr && "op cannot be null");
    assert(start >= 0 && "start must be non-negative");
    assert(length >= 0 && "length must be non-negative");
    assert(start + size <= static_cast<int32_t>(op->getNumOperands()) &&
           "start + size must be less than or equal to the number of operands");
  }

  operator bool() const { return op != nullptr; }

  Operand operator[](int32_t index) const {
    assert(index >= 0 && index < length && "index out of range");
    assert(op != nullptr && "op cannot be null");
    return Operand(op->getOpOperand(start + index));
  }

  template <typename OpTy>
  static MutableOperand get(OpTy op, int32_t index) {
    assert(op != nullptr && "op cannot be null");
    assert(index >= 0 && index < static_cast<int32_t>(op->getNumOperands()) &&
           "index out of range");
    MutableArrayRef<int32_t> segmentSizes = getOperandSegmentSizes(op);
    auto [start, size] = op.getODSOperandIndexAndLength(index);
    int32_t *segment =
        segmentSizes.data() ? segmentSizes.data() + index : nullptr;
    return MutableOperand(op, start, size, segment);
  }

  /// Get the underlying operation.
  Operation *getOp() const { return op; }

  /// Get the operand range.
  OperandRange getRange() const {
    return !op ? OperandRange(nullptr, 0)
               : op->getOperands().slice(start, length);
  }

  /// Get the operand range.
  MutableArrayRef<OpOperand> getMutableRange() const {
    return !op ? MutableArrayRef<OpOperand>()
               : op->getOpOperands().slice(start, length);
  }

  /// Check if the operand range is empty.
  bool empty() const { return length == 0; }

  /// Get the size of the operand range.
  int32_t size() const { return length; }

  /// Assign this range to the given values.
  void assign(ValueRange values);

  /// Append the given values to the range.
  void append(ValueRange values);

  /// Erase the operands within the given sub-range.
  void erase(int32_t subStart, int32_t subLen = 1);

  /// Clear this range and erase all of the operands.
  void clear() { erase(0, length); }

  auto begin() const {
    return llvm::map_iterator(getMutableRange().begin(), toOperand);
  }
  auto end() const {
    return llvm::map_iterator(getMutableRange().end(), toOperand);
  }

private:
  /// Helper function to convert an OpOperand to an Operand.
  static Operand toOperand(OpOperand &operand) { return Operand(operand); }

  /// Update the size of the operand range.
  void updateSize(int32_t newSize);

  Operation *op = nullptr;
  int32_t start = 0;
  int32_t length = 0;
  int32_t *segment = nullptr;
};

} // namespace mlir::aster

#endif // ASTER_IR_OPERAND_H
