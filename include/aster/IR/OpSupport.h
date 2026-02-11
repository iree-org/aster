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

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::aster {
/// Get an operand range from a mutable array of operands.
inline OperandRange
getAsOperandRange(llvm::MutableArrayRef<OpOperand> operands) {
  return OperandRange(operands.data(), operands.size());
}
} // namespace mlir::aster

#endif // ASTER_IR_OPSUPPORT_H
