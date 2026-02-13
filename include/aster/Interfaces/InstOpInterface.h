//===- InstOpInterface.h - InstOp interface ---------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the aster instruction operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACE_H
#define ASTER_INTERFACES_INSTOPINTERFACE_H

#include "aster/IR/OpSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class OpBuilder;
namespace aster {
class InstOpInterface;
/// Operand and result information for the instruction operation.
struct InstOpInfo {
  InstOpInfo(int32_t numLeadingOperands, int32_t numInstOuts,
             int32_t numInstIns, int32_t numLeadingResults,
             int32_t numInstResults)
      : numLeadingOperands(numLeadingOperands), numInstOuts(numInstOuts),
        numInstIns(numInstIns), numLeadingResults(numLeadingResults),
        numInstResults(numInstResults) {}
  const int32_t numLeadingOperands;
  const int32_t numInstOuts;
  const int32_t numInstIns;
  const int32_t numLeadingResults;
  const int32_t numInstResults;
  /// Get the leading operands.
  ValueRange getLeadingOperands(ValueRange operands) {
    return operands.slice(0, numLeadingOperands);
  }
  /// Get the instruction output operands.
  ValueRange getInstOuts(ValueRange operands) {
    return operands.slice(numLeadingOperands, numInstOuts);
  }
  /// Get the instruction input operands.
  ValueRange getInstIns(ValueRange operands) {
    return operands.slice(numLeadingOperands + numInstOuts, numInstIns);
  }
  /// Get the trailing operands.
  ValueRange getTrailingOperands(ValueRange operands) {
    return operands.drop_front(numLeadingOperands + numInstOuts + numInstIns);
  }
  /// Get the leading results.
  ResultRange getLeadingResults(ResultRange results) {
    return results.slice(0, numLeadingResults);
  }
  /// Get the instruction results.
  ResultRange getInstResults(ResultRange results) {
    return results.slice(numLeadingResults, numInstResults);
  }
  /// Get the trailing results.
  ResultRange getTrailingResults(ResultRange results) {
    return results.drop_front(numLeadingResults + numInstResults);
  }
};

namespace detail {
/// Returns the speculatability of the instruction. If the instruction has pure
/// value semantics, this function returns Speculatable. Otherwise, it returns
/// NotSpeculatable.
Speculation::Speculatability getInstSpeculatabilityImpl(InstOpInterface op);

/// Returns the memory effects of the instruction. For each register operand
/// without value semantics this function adds read or write effects in the
/// corresponding resource.
void getInstEffectsImpl(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

/// Verify the instruction operation.
LogicalResult verifyInstImpl(InstOpInterface op);

/// Returns true if all the register operands have value semantics.
bool hasPureValueSemanticsImpl(InstOpInterface op);
struct InstAttrStorage;
} // namespace detail
} // namespace aster
} // namespace mlir

#include "aster/Interfaces/InstOpInterface.h.inc"

#endif // ASTER_INTERFACES_INSTOPINTERFACE_H
