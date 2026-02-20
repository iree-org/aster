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
    ValueRange newOuts, ValueRange newIns, ValueRange operands,
    TypeRange results, int32_t kLeadingOperandsSize, int32_t kOutsSize,
    int32_t kInsSize, int32_t kLeadingResultsSize,
    function_ref<std::pair<unsigned, unsigned>(unsigned)>
        getOperandIndexAndLength,
    function_ref<std::pair<unsigned, unsigned>(unsigned)>
        getResultIndexAndLength,
    SmallVectorImpl<Value> &newOperands, SmallVectorImpl<Type> &newResultTypes,
    MutableArrayRef<int32_t> resultSegmentSizes) {
  // Helper function to get the number of operands.
  auto getNumOperands = [&](int64_t start, int64_t count) {
    int64_t c = 0;
    for (int64_t i = 0; i < count; ++i) {
      auto [_, size] = getOperandIndexAndLength(start + i);
      c += size;
    }
    return c;
  };
  auto getNumResults = [&](int64_t start, int64_t count) {
    int64_t c = 0;
    for (int64_t i = 0; i < count; ++i) {
      auto [_, size] = getResultIndexAndLength(start + i);
      c += size;
    }
    return c;
  };
  newOperands.clear();
  newOperands.reserve(operands.size());

  // Append the leading operands.
  llvm::append_range(
      newOperands, operands.slice(0, getNumOperands(0, kLeadingOperandsSize)));

  // Append the output operands.
  ValueRange outOperands = operands.slice(
      newOperands.size(), getNumOperands(kLeadingOperandsSize, kOutsSize));
  if (outOperands.size() != newOuts.size())
    return failure();
  llvm::append_range(newOperands, newOuts);

  // Append the input operands.
  ValueRange insOperands = operands.slice(
      newOperands.size(),
      getNumOperands(kLeadingOperandsSize + kOutsSize, kInsSize));
  if (insOperands.size() != newIns.size())
    return failure();
  llvm::append_range(newOperands, newIns);

  // Append the trailing operands.
  llvm::append_range(newOperands, operands.drop_front(newOperands.size()));

  newResultTypes.clear();
  // Early exit if the instruction has no output operands.
  if (kOutsSize == 0) {
    llvm::append_range(newResultTypes, results);
    return success();
  }

  // Append the leading results.
  llvm::append_range(newResultTypes,
                     results.slice(0, getNumResults(0, kLeadingResultsSize)));

  // NOTE: this will assert on invalid arguments.
  bool hasSegmentSizes = !resultSegmentSizes.empty();
  if (hasSegmentSizes) {
    resultSegmentSizes =
        resultSegmentSizes.slice(kLeadingResultsSize, kOutsSize);
  }

  assert((!hasSegmentSizes ||
          kOutsSize == static_cast<int64_t>(resultSegmentSizes.size())) &&
         "expected the size of the output operands to be equal to the number "
         "of result segment sizes");

  // Get the trailing results.
  TypeRange trailingResults = results.drop_front(
      newResultTypes.size() + getNumResults(kLeadingResultsSize, kOutsSize));

  // Append the output results.
  for (int64_t out = 0; out < kOutsSize; ++out) {
    auto [start, size] = getOperandIndexAndLength(kLeadingOperandsSize + out);
    ValueRange outOperands = ValueRange(newOperands).slice(start, size);
    int64_t numRes = 0;
    for (Type type : TypeRange(outOperands)) {
      auto regTy = cast<RegisterTypeInterface>(type);
      if (!regTy.hasValueSemantics())
        continue;
      ++numRes;
      newResultTypes.push_back(regTy);
    }
    if (resultSegmentSizes.empty())
      continue;
    resultSegmentSizes[out] = numRes;
  }

  // Append the trailing results.
  llvm::append_range(newResultTypes, trailingResults);
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
