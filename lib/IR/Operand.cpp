//===- Operand.cpp - Aster operand wrapper ------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/IR/Operand.h"

using namespace mlir::aster;

void MutableOperand::updateSize(int32_t newSize) {
  length = newSize;
  if (segment)
    *segment = length;
}

void MutableOperand::assign(ValueRange values) {
  assert(op != nullptr && "op cannot be null");
  op->setOperands(start, length, values);
  updateSize(static_cast<int32_t>(values.size()));
}

void MutableOperand::append(ValueRange values) {
  assert(op != nullptr && "op cannot be null");
  if (values.empty())
    return;
  op->insertOperands(start + length, values);
  updateSize(length + static_cast<int32_t>(values.size()));
}

void MutableOperand::erase(int32_t subStart, int32_t subLen) {
  assert(op != nullptr && "op cannot be null");
  assert(subStart >= 0 && subStart + subLen <= length &&
         "sub-range out of bounds");
  if (empty())
    return;
  op->eraseOperands(start + subStart, subLen);
  updateSize(length - subLen);
}
