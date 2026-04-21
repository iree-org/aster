// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
#define WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"

namespace wave {

// Callback generating a diagnostic.
using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;

class WaveTensorType;

/// Get the hyperparameters from an ancestor operation.
/// Returns nullptr if no hyperparameters are found.
WaveHyperparameterAttr getHyperparameters(mlir::Operation *op);

//-----------------------------------------------------------------------------
// HasWaveIndexMapping trait
//-----------------------------------------------------------------------------

// Common verifier for the optional 'index' attribute used by Wave ops.
mlir::LogicalResult verifyWaveIndexMappings(mlir::Operation *op);

// Trait that checks the 'index' attribute using verifyWaveIndexMappings.
template <typename ConcreteType>
class HasWaveIndexMapping
    : public mlir::OpTrait::TraitBase<ConcreteType, HasWaveIndexMapping> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return verifyWaveIndexMappings(op);
  }
};

mlir::ParseResult parseWaveIndexDict(mlir::OpAsmParser &parser,
                                     mlir::ArrayAttr &out);
void printWaveIndexDict(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                        mlir::ArrayAttr arr);

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface and implementation traits
//-----------------------------------------------------------------------------

namespace detail {
// Propagate shape information from `from` tensor types to `to` tensor types.
// Expects all fully-specified tensor types to have the same shape, prints an
// error message to `errs` otherwise. Update all under-specified tensor types in
// `to` to be fully specified with the shapes extracted from from. For
// fully-specified types in `to`, check if their shapes match those in `from`
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If there was an error
// due to mismatching/irreconcilable types and an error was printed, returns
// `failure`. Otherwise returns an change indicator.
llvm::FailureOr<mlir::ChangeResult>
identityTypeInferencePropagate(llvm::ArrayRef<wave::WaveTensorType> from,
                               llvm::MutableArrayRef<wave::WaveTensorType> to,
                               llvm::StringRef fromName, llvm::StringRef toName,
                               llvm::raw_ostream &errs);
llvm::FailureOr<mlir::ChangeResult>
propagateShapeInformation(wave::WaveTensorType from, wave::WaveTensorType &to,
                          llvm::StringRef fromName, llvm::StringRef toName,
                          llvm::raw_ostream &errs);
llvm::FailureOr<mlir::ChangeResult>
propagateShapeInformation(llvm::ArrayRef<wave::WaveSymbolAttr> from,
                          wave::WaveTensorType &to, llvm::StringRef fromName,
                          llvm::StringRef toName, llvm::raw_ostream &errs);

// Propagate shape information from `source` to `target` and drop the `n`
// `source` dims. Expects both to be fully-specified tensor types. If
// propagation discovers a type conflict, prints the error message to the
// `errs` stream and returns failure. Otherwise returns a tag indicating
// whether the target type changed.
llvm::FailureOr<mlir::ChangeResult> propagateShapeDropTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    llvm::StringRef sourceName, llvm::StringRef targetName, unsigned n,
    llvm::raw_ostream &errs);

// Propagate shape information from `source` to `target` and add `n` trailing
// dims. Expects both to be fully-specified tensor types. If propagation
// discovers a type conflict, prints the error message to the `errs` stream and
// returns failure. Otherwise returns a tag indicating whether the target type
// changed.
llvm::FailureOr<mlir::ChangeResult> propagateShapeAddTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    llvm::StringRef sourceName, llvm::StringRef targetName,
    llvm::ArrayRef<wave::WaveSymbolAttr> newDims, llvm::raw_ostream &errs);

// Propagate type information for reduction operations from operands to results.
// If init is present, we can propagate from it directly, otherwise propagate
// from input after removing the reduction axis.
llvm::FailureOr<mlir::ChangeResult> propagateReductionTypesForward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum, llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs);

// Propagate type information for reduction operations from results to operands.
// Propagates from result to init operand, and "sideways" from input to init
// operand.
llvm::FailureOr<mlir::ChangeResult> propagateReductionTypesBackward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum,
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs);

// Return true if type inference for operands and results of a reduction
// operation is complete, i.e., all values have fully specified types.
bool isReductionTypeInferenceComplete(mlir::Value input, mlir::Value init,
                                      mlir::Value result);

// Check whether the `from` and `to` tensor types have reconcilable shapes and
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If types are
// reconcilable, returns an indicator whether the `to` type will have to be
// updated.
llvm::FailureOr<mlir::ChangeResult>
checkPropagateShapeConflict(wave::WaveTensorType from, wave::WaveTensorType to,
                            llvm::StringRef fromName, llvm::StringRef toName,
                            llvm::raw_ostream &errs);

} // namespace detail

// A trait providing an implementation of the WaveInferTypeOpInterface where
// shapes are propagated from all operands to all results and back as is.
template <typename OpTy>
class IdentityTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, IdentityTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        operandTypes, resultTypes, "operands", "results", errs);
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        resultTypes, operandTypes, "results", "operands", errs);
  }

  llvm::LogicalResult finalizeTypeInference() { return llvm::success(); }
};

// A trait providing an implementation of the WaveInferTypeOpInterface for
// reduction operations. It handles addition/removal of the reduction axis from
// the types. Expects the operation to:
// - have an 'axis' attribute of type WaveSymbolAttr indicating the reduction
//   axis;
// - have 'init' and 'input' operands.
template <typename OpTy>
class ReductionTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, ReductionTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    auto concrete = llvm::cast<OpTy>(this->getOperation());
    wave::WaveSymbolAttr axis = concrete.getReducedSymbol();
    unsigned initOperandNum = concrete.getInitMutable().getOperandNumber();
    // Use the first input for type propagation.
    unsigned inputOperandNum = concrete.getInputs().getBeginOperandIndex();
    return detail::propagateReductionTypesForward(
        axis, initOperandNum, inputOperandNum, operandTypes, resultTypes, errs);
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    auto concrete = llvm::cast<OpTy>(this->getOperation());
    wave::WaveSymbolAttr axis = concrete.getReducedSymbol();
    unsigned initOperandNum = concrete.getInitMutable().getOperandNumber();
    // Use the first input for type propagation.
    unsigned inputOperandNum = concrete.getInputs().getBeginOperandIndex();
    return detail::propagateReductionTypesBackward(
        axis, initOperandNum, inputOperandNum, operandTypes, resultTypes, errs);
  }

  llvm::LogicalResult finalizeTypeInference() {
    auto concrete = llvm::cast<OpTy>(this->getOperation());
    if (detail::isReductionTypeInferenceComplete(concrete.getInputs().front(),
                                                 concrete.getInit(),
                                                 concrete.getResult()))
      concrete.removeAxisAttr();
    return llvm::success();
  }
};

// A trait providing an implementation of the WaveInferTypeOpInterface where no
// shape propagation is needed. E.g. for operations that only have operands and
// no results.
template <typename OpTy>
class NoOpTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, NoOpTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }

  llvm::LogicalResult finalizeTypeInference() { return llvm::success(); }
};

// Verify that element types of Wave tensors or vectors match between LHS and
// RHS. Emit diagnostic errors and return a failure when it is not the case.
namespace detail {
llvm::LogicalResult verifyElementTypesMatch(std::optional<mlir::Location> loc,
                                            llvm::StringRef lhsName,
                                            mlir::Type lhs,
                                            llvm::StringRef rhsName,
                                            mlir::Type rhs);

// Verify if two Wave tensor or vector types are compatible:
//   - their element types are equal unless `includeElementalType` is false;
//   - their address spaces are equal unless `includeAddressSpace` is false;
//   - tensor symbolic shapes are either equal or at least one of them is
//     underspecified;
//   - tensor address spaces are either equal or at least one of them is
//     underspecified;
// When it is not the case, return failure and optionally report an error if a
// location is provided.
llvm::LogicalResult verifyTypesCompatible(
    mlir::Type lhs, mlir::Type rhs, bool includeAddressSpace,
    bool includeElementalType,
    std::optional<mlir::Location> errorLocation = std::nullopt,
    llvm::StringRef lhsName = "", llvm::StringRef rhsName = "");

// Verify that the shapes of two Wave tensor types are compatible, i.e., they
// have the same rank and the corresponding dimensions are equal. Emit
// diagnostic errors and return failure when it is not the case.
llvm::LogicalResult
verifyTensorShapesCompatible(wave::WaveTensorType lhs, wave::WaveTensorType rhs,
                             std::optional<mlir::Location> errorLocation,
                             llvm::StringRef lhsName, llvm::StringRef rhsName);

// Verify that specified dimensions match between LHS and RHS, the lists of
// dimensions are expected to be co-indexed. Emit diagnostic errors and
// return failure when it is not the case.
llvm::LogicalResult
verifyTypesMatchingDimensions(std::optional<mlir::Location> loc,
                              llvm::StringRef lhsName, wave::WaveTensorType lhs,
                              llvm::ArrayRef<int> lhsDims,
                              llvm::StringRef rhsName, wave::WaveTensorType rhs,
                              llvm::ArrayRef<int> rhsDims);

// Verification logic for the compatible-operands traits. Succeeds if all wave
// tensor-typed operands and results have compatible shapes and, if the
// corresponding flag is set, compatible address spaces.
llvm::LogicalResult verifyCompatibleOperandsAndResultsOpTrait(
    mlir::Operation *op, bool includeAddressSpace, bool includeElementalType);
}; // namespace detail

template <typename OpTy>
class CompatibleOperandsAndResultsOpTrait
    : public mlir::OpTrait::TraitBase<OpTy,
                                      CompatibleOperandsAndResultsOpTrait> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyCompatibleOperandsAndResultsOpTrait(
        op, /*includeAddressSpace=*/true, /*includeElementalType=*/true);
  }
};

template <typename OpTy>
class CompatibleOperandsAndResultsIgnoreSpaceOpTrait
    : public mlir::OpTrait::TraitBase<
          OpTy, CompatibleOperandsAndResultsIgnoreSpaceOpTrait> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyCompatibleOperandsAndResultsOpTrait(
        op, /*includeAddressSpace=*/false, /*includeElementalType=*/true);
  }
};

template <typename OpTy>
class CompatibleOperandsAndResultsShapeOpTrait
    : public mlir::OpTrait::TraitBase<
          OpTy, CompatibleOperandsAndResultsShapeOpTrait> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyCompatibleOperandsAndResultsOpTrait(
        op, /*includeAddressSpace=*/true, /*includeElementalType=*/false);
  }
};

// ----------------------------------------------------------------------------
// Reduction operation traits
// ----------------------------------------------------------------------------

namespace detail {
// Return the symbol along which the reduction happens if known given the axis
// and the input type.
WaveSymbolAttr getReducedSymbol(mlir::Operation *op, WaveSymbolAttr axisAttr,
                                mlir::Type inputType);

// Verify the types of a reduction operation.
llvm::LogicalResult verifyReductionOperation(mlir::Operation *op,
                                             mlir::Type inputType,
                                             mlir::Type initType,
                                             mlir::Type resultType,
                                             mlir::Attribute axisAttr);

// Return the symbol along which the reduction happens if known.
template <typename OpTy>
static inline WaveSymbolAttr getReducedSymbol(OpTy op) {
  return wave::detail::getReducedSymbol(op, op.getAxisAttr(),
                                        op.getInputs().front().getType());
}

// Common verification logic for reduction operations. All inputs must have the
// same type; we verify against the first input.
template <typename OpTy>
static inline llvm::LogicalResult verifyReductionOperation(OpTy op) {
  if (op.getInputs().empty())
    return op.emitOpError("expected at least one input");
  mlir::Type firstInputType = op.getInputs().front().getType();
  for (mlir::Value input : op.getInputs().drop_front()) {
    if (input.getType() != firstInputType)
      return op.emitOpError() << "all inputs must have the same type, but got "
                              << firstInputType << " and " << input.getType();
  }
  return wave::detail::verifyReductionOperation(
      op, firstInputType, op.getInit().getType(), op.getResult().getType(),
      op.getAxisAttr());
}
} // namespace detail

template <typename OpTy>
class WaveReductionOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, WaveReductionOpTrait> {
public:
  ::wave::WaveSymbolAttr getReducedSymbol() {
    return detail::getReducedSymbol(llvm::cast<OpTy>(this->getOperation()));
  }

  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyReductionOperation(llvm::cast<OpTy>(op));
  }
};

} // namespace wave

#include "water/Dialect/Wave/IR/WaveOpInterfaces.h.inc"

#endif // WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
