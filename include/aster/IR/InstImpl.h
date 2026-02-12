//===- InstImpl.h - Instruction Implementation ------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares instruction implementation support for ASTER.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_INSTIMPL_H
#define ASTER_IR_INSTIMPL_H

#include "aster/IR/OpSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"

namespace mlir::aster {
namespace detail {
/// Infer the types of the instruction.
template <typename OpTy>
inline LogicalResult
inferTypesImpl(ValueRange operands, DictionaryAttr attrs,
               OpaqueProperties properties, RegionRange regions,
               SmallVectorImpl<Type> &inferredReturnTypes) {
  typename OpTy::Adaptor adaptor(operands, attrs, properties, regions);
  int64_t numLeadingOperands = OpTy::getNumLeadingOperands(adaptor);
  int64_t numInstOuts = OpTy::getNumInstOuts(adaptor);
  llvm::append_range(
      inferredReturnTypes,
      TypeRange(operands.slice(numLeadingOperands, numInstOuts)));
  return success();
}
} // namespace detail

/// Trait to provide utility methods for instruction operations.
template <typename ConcreteType>
struct InstTrait : public OpTrait::TraitBase<ConcreteType, InstTrait> {
  /// Get the number of leading operands.
  /// NOTE: this method is templated to allow it working with the Op adator.
  template <typename OpTy>
  static int64_t getNumLeadingOperands(OpTy op) {
    int64_t c = 0;
    for (int64_t i = 0; i < ConcreteType::kLeadingOperandsSize; ++i) {
      auto [start, size] = op.getODSOperandIndexAndLength(i);
      c += size;
    }
    return c;
  }

  /// Get the number of instruction output operands.
  /// NOTE: this method is templated to allow it working with the Op adator.
  template <typename OpTy>
  static int64_t getNumInstOuts(OpTy op) {
    int64_t c = 0;
    for (int64_t i = 0; i < ConcreteType::kOutsSize; ++i) {
      auto [start, size] = op.getODSOperandIndexAndLength(
          i + ConcreteType::kLeadingOperandsSize);
      c += size;
    }
    return c;
  }

  /// Get the number of instruction input operands.
  static int64_t getNumInstIns(ConcreteType op) {
    int64_t c = 0;
    int64_t startPos =
        ConcreteType::kLeadingOperandsSize + ConcreteType::kOutsSize;
    for (int64_t i = 0; i < ConcreteType::kInsSize; ++i) {
      auto [start, size] = op.getODSOperandIndexAndLength(i + startPos);
      c += size;
    }
    return c;
  }

  /// Get the instruction output operands.
  static OperandRange getInstOutsImpl(ConcreteType op) {
    MutableArrayRef<OpOperand> operands = op.getOperation()->getOpOperands();
    return getAsOperandRange(operands).slice(getNumLeadingOperands(op),
                                             getNumInstOuts(op));
  }

  /// Get the instruction input operands.
  static OperandRange getInstInsImpl(ConcreteType op) {
    MutableArrayRef<OpOperand> operands = op.getOperation()->getOpOperands();
    return getAsOperandRange(operands).slice(getNumInstOuts(op),
                                             getNumInstIns(op));
  }

  /// Get the number of leading results.
  static int64_t getNumLeadingResults(ConcreteType op) {
    int64_t c = 0;
    for (int64_t i = 0; i < ConcreteType::kLeadingResultsSize; ++i) {
      auto [start, size] = op.getODSResultIndexAndLength(i);
      c += size;
    }
    return c;
  }

  /// Get the number of instruction results.
  static int64_t getNumInstResults(ConcreteType op) {
    int64_t c = 0;
    int64_t startPos = ConcreteType::kLeadingResultsSize;
    for (int64_t i = 0; i < ConcreteType::kOutsSize; ++i) {
      auto [start, size] = op.getODSResultIndexAndLength(i + startPos);
      c += size;
    }
    return c;
  }

  /// Get the instruction results.
  static ResultRange getInstResultsImpl(ConcreteType op) {
    ResultRange results = op.getOperation()->getResults();
    return results.slice(getNumLeadingResults(op), getNumInstResults(op));
  }
};
} // namespace mlir::aster

#endif // ASTER_IR_INSTIMPL_H
