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
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::aster {
namespace detail {
/// Infer the types of the instruction.
LogicalResult inferTypesImpl(TypeRange outTypes,
                             SmallVectorImpl<Type> &inferredReturnTypes,
                             ArrayRef<int32_t> outsSegmentSizes,
                             MutableArrayRef<int32_t> resultOutSegmentSizes);
template <typename OpTy>
inline LogicalResult
inferTypesImpl(ValueRange operands, DictionaryAttr attrs,
               OpaqueProperties properties, RegionRange regions,
               SmallVectorImpl<Type> &inferredReturnTypes) {
  // This function only supports instructions with no leading results.
  static_assert(
      (OpTy::kLeadingResultsSize == 0 && OpTy::kTrailingResultsSize == 0),
      "expected leading and trailing results size to be 0");

  // Get the number of leading operands and instruction output operands.
  typename OpTy::Adaptor adaptor(operands, attrs, properties, regions);
  int64_t numLeadingOperands = OpTy::getNumLeadingOperands(adaptor);
  int64_t numInstOuts = OpTy::getNumInstOuts(adaptor);

  // Get the operand and result segment sizes.
  SmallVector<int32_t> outsSegmentSizes;
  MutableArrayRef<int32_t> resultOutSegmentSizes;
  if constexpr (OpTy::kOutsSize > 1) {
    // Get the out operand segment sizes.
    for (int64_t i = 0; i < OpTy::kOutsSize; ++i) {
      auto [start, size] =
          adaptor.getODSOperandIndexAndLength(i + OpTy::kLeadingOperandsSize);
      outsSegmentSizes.push_back(size);
    }
    resultOutSegmentSizes = getResultSegmentSizes<OpTy>(properties);
    resultOutSegmentSizes = resultOutSegmentSizes.slice(0, OpTy::kOutsSize);
  }
  return inferTypesImpl(
      TypeRange(operands.slice(numLeadingOperands, numInstOuts)),
      inferredReturnTypes, outsSegmentSizes, resultOutSegmentSizes);
}

/// Add write effects to the given effects for the given address.
void getWriteEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

/// Add read effects to the given effects for the given address.
void getReadEffectsImpl(
    OpOperand &addr,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

/// Populates newOperands and newResultTypes with the new operands and results.
/// Also mutates the result segment sizes if needed.
LogicalResult cloneInstOperandsResultsImpl(
    ValueRange newOuts, ValueRange newIns, ValueRange operands,
    TypeRange results, int32_t kLeadingOperandsSize, int32_t kOutsSize,
    int32_t kInsSize, int32_t kLeadingResultsSize,
    function_ref<std::pair<unsigned, unsigned>(unsigned)>
        getOperandIndexAndLength,
    function_ref<std::pair<unsigned, unsigned>(unsigned)>
        getResultIndexAndLength,
    SmallVectorImpl<Value> &newOperands, SmallVectorImpl<Type> &newResultTypes,
    MutableArrayRef<int32_t> resultSegmentSizes);

/// Clones the instruction operation with new operands and results.
template <typename OpTy>
OpTy cloneInstImpl(OpTy op, OpBuilder &builder, ValueRange outs,
                   ValueRange ins) {

  auto getOperandIndexAndLength = [&](unsigned index) {
    return op.getODSOperandIndexAndLength(index);
  };
  auto getResultIndexAndLength = [&](unsigned index) {
    return op.getODSResultIndexAndLength(index);
  };
  // Get the operation's attirbutes.
  SmallVector<NamedAttribute> attributes =
      llvm::to_vector(op->getDiscardableAttrs());
  // Copy the operation's properties.
  typename OpTy::Properties properties = op.getProperties();
  // Get the new operands and result types.
  SmallVector<Value> operands;
  SmallVector<Type> resultTypes;
  LogicalResult result = cloneInstOperandsResultsImpl(
      outs, ins, op->getOperands(), TypeRange(op->getResults()),
      OpTy::kLeadingOperandsSize, OpTy::kOutsSize, OpTy::kInsSize,
      OpTy::kLeadingResultsSize, getOperandIndexAndLength,
      getResultIndexAndLength, operands, resultTypes,
      getResultSegmentSizes<OpTy>(OpaqueProperties(&properties)));
  if (failed(result))
    return nullptr;
  return OpTy::create(builder, op.getLoc(), resultTypes, operands, properties,
                      attributes);
}

/// Computes liveness transfer function for the instruction.
LogicalResult livenessTransferFunctionImpl(InstOpInterface op,
                                           LivenessCallback insertCallback,
                                           LivenessCallback removeCallback,
                                           IsLiveCallback isLiveCallback);
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

  /// Get the operand and result information for the instruction.
  static InstOpInfo getInstInfoImpl(ConcreteType op) {
    return InstOpInfo(getNumLeadingOperands(op), getNumInstOuts(op),
                      getNumInstIns(op), getNumLeadingResults(op),
                      getNumInstResults(op));
  }

  /// Clones the instruction operation with new operands and results.
  InstOpInterface cloneInst(OpBuilder &builder, ValueRange outs,
                            ValueRange ins) {
    return ::mlir::aster::detail::cloneInstImpl<ConcreteType>(
        static_cast<ConcreteType &>(*this), builder, outs, ins);
  }
};

/// Forward iterator for co-iterating over the instruction output operands and
/// results.
struct TiedInstOutsIterator {
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::pair<Value, OpResult>;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = value_type;
  TiedInstOutsIterator(int64_t numOuts, ValueRange::iterator operandIt,
                       ResultRange::iterator resultIt)
      : numOuts(numOuts), operandIt(operandIt), resultIt(resultIt) {}

  /// Returns the current operand and result. If the operand does not have value
  /// semantics, the result is nullptr.
  reference operator*() const {
    auto regTy = cast<RegisterTypeInterface>((*operandIt).getType());
    if (!regTy.hasValueSemantics())
      return std::make_pair(*operandIt, nullptr);
    return std::make_pair(*operandIt, *resultIt);
  }

  /// Advances the iterator to the next operand and result.
  TiedInstOutsIterator &operator++() {
    if (numOuts <= operandIt.getIndex())
      return *this;
    if (auto regTy = cast<RegisterTypeInterface>((*operandIt).getType());
        regTy.hasValueSemantics())
      ++resultIt;
    ++operandIt;
    return *this;
  }

  TiedInstOutsIterator operator++(int) {
    TiedInstOutsIterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool operator==(const TiedInstOutsIterator &other) const {
    return operandIt == other.operandIt;
  }
  bool operator!=(const TiedInstOutsIterator &other) const {
    return !(*this == other);
  }

  size_t size() const { return numOuts; }

private:
  int64_t numOuts;
  ValueRange::iterator operandIt;
  ResultRange::iterator resultIt;
};

/// Range for co-iterating over the instruction output operands and results.
struct TiedInstOutsRange {
  using iterator = TiedInstOutsIterator;
  TiedInstOutsRange(TiedInstOutsRange &&other) noexcept
      : outs(std::move(other.outs)), results(std::move(other.results)) {}
  TiedInstOutsRange(InstOpInterface instOp)
      : outs(instOp.getInstOuts()), results(instOp.getInstResults()) {}
  iterator begin() {
    return iterator(outs.size(), outs.begin(), results.begin());
  }
  iterator end() { return iterator(outs.size(), outs.end(), results.end()); }

  size_t size() const { return outs.size(); }

private:
  ValueRange outs;
  ResultRange results;
};

/// ADL-friendly begin/end for use with llvm::zip_equal and other ADL-based
/// range utilities.
inline TiedInstOutsRange::iterator begin(TiedInstOutsRange &r) {
  return r.begin();
}
inline TiedInstOutsRange::iterator end(TiedInstOutsRange &r) { return r.end(); }

/// Get all allocas behind a value, using the following rules:
/// - If the value is not a register, it returns an empty range.
/// - If the value is an alloca, it returns the alloca.
/// - If value is an operand bundle without value semantics it checks the inputs
/// and if all the inputs are alloca, it returns them.
/// - Otherwise, it returns failure.
FailureOr<ValueRange> getAllocasOrFailure(Value value);

/// Similar to `getAllocasOrFailure` but for a range of values.
inline LogicalResult getAllocasOrFailure(ValueRange values,
                                         SmallVectorImpl<Value> &allocas) {
  for (Value value : values) {
    FailureOr<ValueRange> result = getAllocasOrFailure(value);
    if (failed(result))
      return failure();
    llvm::append_range(allocas, *result);
  }
  return success();
}
} // namespace mlir::aster

#endif // ASTER_IR_INSTIMPL_H
