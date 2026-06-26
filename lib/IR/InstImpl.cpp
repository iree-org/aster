//===- InstImpl.cpp - Instruction Implementation ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/IR/InstImpl.h"
#include "aster/Interfaces/AllocaOpInterface.h"
#include "aster/Interfaces/OperandBundleOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::aster;

void aster::detail::getWriteEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &addr);
}

void aster::detail::getReadEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &addr);
}

LogicalResult
aster::detail::inferTypesImpl(TypeRange outTypes,
                              SmallVectorImpl<Type> &inferredReturnTypes,
                              ArrayRef<int32_t> outsSegmentSizes,
                              MutableArrayRef<int32_t> resultOutSegmentSizes) {
  llvm::append_range(inferredReturnTypes,
                     llvm::make_filter_range(outTypes, [](Type type) {
                       auto regType = dyn_cast<RegisterTypeInterface>(type);
                       return regType && regType.hasValueSemantics();
                     }));
  assert(outsSegmentSizes.empty() == resultOutSegmentSizes.empty() &&
         "expected both or neither of outsSegmentSizes and "
         "resultOutSegmentSizes to be present");
  if (resultOutSegmentSizes.empty())
    return success();
  int64_t pos = 0;
  for (auto &&[outSegmentSize, resultOutSegmentSize] :
       llvm::zip_equal(outsSegmentSizes, resultOutSegmentSizes)) {
    resultOutSegmentSize = 0;
    for (int64_t i = 0; i < outSegmentSize; ++i) {
      auto type = cast<RegisterTypeInterface>(outTypes[pos++]);
      if (type.hasValueSemantics())
        ++resultOutSegmentSize;
    }
  }
  return success();
}

LogicalResult aster::detail::cloneInstOperandsResultsImpl(
    InstOpInfo info, OperandRange operands, ResultRange results,
    ValueRange newOuts, ValueRange newIns, ArrayRef<int32_t> outsSegmentSizes,
    MutableArrayRef<int32_t> resultOutSegmentSizes,
    SmallVectorImpl<Value> &newOperands,
    SmallVectorImpl<Type> &newResultTypes) {
  // The caller supplies one value per out and in operand. Absent optionals are
  // already excluded from the InstOpInfo counts, so the slices stay valid.
  if (info.getInstOuts(operands).size() != newOuts.size() ||
      info.getInstIns(operands).size() != newIns.size())
    return failure();

  // Operands: [leading | newOuts | newIns | trailing].
  newOperands.clear();
  newOperands.reserve(operands.size());
  llvm::append_range(newOperands, info.getLeadingOperands(operands));
  llvm::append_range(newOperands, newOuts);
  llvm::append_range(newOperands, newIns);
  llvm::append_range(newOperands, info.getTrailingOperands(operands));

  // Result types: [leading | out results | trailing].
  // An out result exists only for a value-semantic out operand, so moving an
  // out to storage drops its result.
  // `inferTypesImpl` appends the value-semantic-filtered out types and, when
  // segment arrays are present, recomputes the per-out-group result-segment
  // sizes in place.
  newResultTypes.clear();
  llvm::append_range(newResultTypes,
                     info.getLeadingResults(results).getTypes());
  assert(
      (outsSegmentSizes.empty() ||
       std::accumulate(outsSegmentSizes.begin(), outsSegmentSizes.end(),
                       int64_t{0}) == static_cast<int64_t>(newOuts.size())) &&
      "out segment sizes inconsistent with the new output operand count");
  if (failed(inferTypesImpl(TypeRange(newOuts), newResultTypes,
                            outsSegmentSizes, resultOutSegmentSizes)))
    return failure();
  llvm::append_range(newResultTypes,
                     info.getTrailingResults(results).getTypes());
  return success();
}

FailureOr<ValueRange> mlir::aster::getAllocasOrFailure(Value value) {
  // If the value is not a register, return an empty range.
  if (!isa<RegisterTypeInterface>(value.getType()))
    return ValueRange{};

  // Handle alloca operations.
  if (auto allocaOp = value.getDefiningOp<AllocaOpInterface>())
    return allocaOp->getResults();

  // Handle operand bundle operations.
  if (auto bundleOp = value.getDefiningOp<OperandBundleOpInterface>()) {
    ValueRange result = bundleOp.unpackBundle();
    if (!llvm::all_of(result, [](Value value) {
          auto allocaOp = value.getDefiningOp<AllocaOpInterface>();
          if (!allocaOp)
            return false;
          auto regTy =
              dyn_cast<RegisterTypeInterface>(allocaOp.getAlloca().getType());
          return regTy && !regTy.hasValueSemantics();
        }))
      return failure();
    return result;
  }

  // Fail in all other cases.
  return failure();
}

/// For each value, call back with either the value itself (if it has value
/// semantics) or its underlying allocas. Returns failure if a
/// non-value-semantic value cannot be resolved to allocas.
static LogicalResult forEachLivenessValue(ValueRange values,
                                          LivenessCallback callback) {
  for (Value value : values) {
    if (auto regTy = dyn_cast<RegisterTypeInterface>(value.getType());
        regTy && regTy.hasValueSemantics()) {
      callback(value);
      continue;
    }
    FailureOr<ValueRange> allocas = getAllocasOrFailure(value);
    if (failed(allocas))
      return failure();
    callback(*allocas);
  }
  return mlir::success();
}

LogicalResult mlir::aster::detail::livenessTransferFunctionImpl(
    InstOpInterface op, LivenessCallback insertCallback,
    LivenessCallback removeCallback, IsLiveCallback isLiveCallback) {
  // Def: for each tied (out, result) pair, remove the defined value.
  SmallVector<Value, 4> definedValues;
  for (auto [out, res] : TiedInstOutsRange(op))
    definedValues.push_back(res ? res : out);
  if (failed(forEachLivenessValue(definedValues, removeCallback)))
    return failure();
  // Use: for each ins operand, insert as live.
  return forEachLivenessValue(op.getInstIns(), insertCallback);
}
