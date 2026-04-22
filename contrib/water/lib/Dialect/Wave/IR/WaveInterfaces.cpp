// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;

//-----------------------------------------------------------------------------
// getHyperparameters
//-----------------------------------------------------------------------------

wave::WaveHyperparameterAttr wave::getHyperparameters(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto hyperparams = current->getAttrOfType<WaveHyperparameterAttr>(
            WaveDialect::kHyperparameterAttrName))
      return hyperparams;
  }
  return nullptr;
}

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface helpers
//-----------------------------------------------------------------------------

// Check whether the shape of the `to` tensor is reconcilable with the shape
// provided in the `from` array and print error messages to errs otherwise.
static FailureOr<ChangeResult>
checkPropagateShapeConflict(ArrayRef<wave::WaveSymbolAttr> from,
                            wave::WaveTensorType to, llvm::StringRef fromName,
                            llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!to || from == to.getShape())
    return ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "(";
  llvm::interleaveComma(from, errs);
  errs << ") to " << toName << "(" << to << ")";
  return failure();
}

llvm::FailureOr<ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from)
    return ChangeResult::NoChange;

  FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      from.getShape(), to, fromName, toName, llvm::nulls());
  if (succeeded(res))
    return res;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return failure();
}

llvm::FailureOr<ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !from.getFullySpecified())
    return ChangeResult::NoChange;
  llvm::FailureOr<ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return ChangeResult::Change;
}

FailureOr<ChangeResult> wave::detail::propagateShapeInformation(
    ArrayRef<wave::WaveSymbolAttr> from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<ChangeResult> res =
      ::checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return ChangeResult::Change;
}

llvm::FailureOr<ChangeResult> wave::detail::identityTypeInferencePropagate(
    llvm::ArrayRef<wave::WaveTensorType> from,
    llvm::MutableArrayRef<wave::WaveTensorType> to, llvm::StringRef fromName,
    llvm::StringRef toName, llvm::raw_ostream &errs) {
  auto it = llvm::find_if(from, [](wave::WaveTensorType type) {
    return type && type.getFullySpecified();
  });
  if (it == from.end())
    return ChangeResult::NoChange;

  // Expect all fully-specified "from" types to have the same shape.
  for (auto [i, fr] : llvm::enumerate(from)) {
    llvm::FailureOr<ChangeResult> res =
        checkPropagateShapeConflict(*it, fr, fromName, toName, errs);
    if (failed(res)) {
      errs << " for " << fromName << " #" << i;
      return res;
    }
  }

  ChangeResult changeResult = ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    llvm::FailureOr<ChangeResult> res =
        propagateShapeInformation(*it, toType, fromName, toName, errs);
    if (failed(res)) {
      errs << " for " << fromName << " #" << i;
      return failure();
    }

    changeResult |= *res;
  }
  return changeResult;
}

// Propagate type information from the reduction input type by removing the
// reduction axis from it to the given type.
static FailureOr<ChangeResult>
propagateFromReductionInput(wave::WaveTensorType inputType,
                            wave::WaveSymbolAttr axis, wave::WaveTensorType &to,
                            StringRef toName, raw_ostream &errs) {
  if (!inputType || !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  SmallVector<wave::WaveSymbolAttr> filteredShape = llvm::filter_to_vector(
      inputType.getShape(),
      [&](wave::WaveSymbolAttr dim) { return dim != axis; });
  assert(inputType.getRank() - 1 == filteredShape.size() &&
         "expected rank to be reduced by 1 in reduction");
  auto inferredType = wave::WaveTensorType::get(
      inputType.getContext(), filteredShape, /*fully_specified=*/true,
      inputType.getElementType(), inputType.getAddressSpace());

  return wave::detail::propagateShapeInformation(inferredType, to, "input",
                                                 toName, errs);
}

FailureOr<ChangeResult> wave::detail::propagateShapeDropTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    StringRef sourceName, StringRef targetName, unsigned n,
    llvm::raw_ostream &errs) {
  if (!source || !source.getFullySpecified())
    return ChangeResult::NoChange;

  ArrayRef<wave::WaveSymbolAttr> expectedShape = source.getShape().drop_back(n);
  FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      expectedShape, target, sourceName, targetName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;

  target = target.copyShapeFrom(expectedShape);
  return ChangeResult::Change;
}

FailureOr<ChangeResult> wave::detail::propagateShapeAddTrailingDims(
    wave::WaveTensorType source, wave::WaveTensorType &target,
    StringRef sourceName, StringRef targetName,
    llvm::ArrayRef<wave::WaveSymbolAttr> newDims, llvm::raw_ostream &errs) {
  if (!source || !source.getFullySpecified())
    return ChangeResult::NoChange;

  SmallVector<wave::WaveSymbolAttr> resultShape(source.getShape());
  llvm::append_range(resultShape, newDims);
  llvm::FailureOr<ChangeResult> res = ::checkPropagateShapeConflict(
      resultShape, target, sourceName, targetName, errs);
  if (failed(res) || *res == ChangeResult::NoChange)
    return res;
  target = target.copyShapeFrom(resultShape);
  return ChangeResult::Change;
}

llvm::FailureOr<ChangeResult> wave::detail::propagateReductionTypesForward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum, llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  // If init is present, we can propagate from it directly,
  // otherwise propagate from input after removing the axis.
  FailureOr<ChangeResult> maybeChangeResult =
      wave::detail::propagateShapeInformation(
          operandTypes[initOperandNum], resultTypes[0], "init", "result", errs);
  if (failed(maybeChangeResult))
    return failure();

  wave::WaveTensorType inputType = operandTypes[inputOperandNum];
  maybeChangeResult =
      maybeChangeResult | propagateFromReductionInput(
                              inputType, axis, resultTypes[0], "result", errs);
  maybeChangeResult = maybeChangeResult | propagateShapeDropTrailingDims(
                                              inputType, resultTypes[0],
                                              "input", "result", 1, errs);
  return maybeChangeResult;
}

llvm::FailureOr<ChangeResult> wave::detail::propagateReductionTypesBackward(
    wave::WaveSymbolAttr axis, unsigned initOperandNum,
    unsigned inputOperandNum,
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  FailureOr<ChangeResult> maybeChangeResult =
      wave::detail::propagateShapeInformation(
          resultTypes[0], operandTypes[initOperandNum], "result", "init", errs);
  if (failed(maybeChangeResult))
    return failure();

  // Propagate "sideways" from input to init operand.
  wave::WaveTensorType inputType = operandTypes[inputOperandNum];
  maybeChangeResult =
      maybeChangeResult |
      propagateFromReductionInput(inputType, axis, operandTypes[initOperandNum],
                                  "init", errs);

  // Since we only reduce trailing dimensions, we can infer the operand shape by
  // adding the reduction axis back to the result.
  maybeChangeResult =
      maybeChangeResult | propagateShapeAddTrailingDims(
                              resultTypes[0], operandTypes[inputOperandNum],
                              "result", "input", {axis}, errs);

  return maybeChangeResult;
}

bool wave::detail::isReductionTypeInferenceComplete(Value input, Value init,
                                                    Value result) {
  return llvm::all_of(
      llvm::ArrayRef<Value>{input, init, result}, [&](Value value) {
        return llvm::cast<WaveTensorType>(value.getType()).getFullySpecified();
      });
}

//-----------------------------------------------------------------------------
// Verification helpers
//-----------------------------------------------------------------------------

// Update negative indices in the array to positive equivalents given the total
// rank.
static void updateNegativeIndices(llvm::MutableArrayRef<int> indices,
                                  int rank) {
  for (int &index : indices) {
    if (index < 0)
      index += rank;
  }
}

llvm::LogicalResult wave::detail::verifyTypesMatchingDimensions(
    std::optional<Location> loc, llvm::StringRef lhsName,
    wave::WaveTensorType lhs, llvm::ArrayRef<int> lhsDims,
    llvm::StringRef rhsName, wave::WaveTensorType rhs,
    llvm::ArrayRef<int> rhsDims) {
  assert(lhsDims.size() == rhsDims.size() &&
         "expected lhs and rhs dim lists to be co-indexed");

  // Under-specified types are okay everywhere.
  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return success();

  llvm::SmallVector<int> lhsDimsVec(lhsDims), rhsDimsVec(rhsDims);
  updateNegativeIndices(lhsDimsVec, lhs.getRank());
  updateNegativeIndices(rhsDimsVec, rhs.getRank());
  for (auto &&[lhsDim, rhsDim] : llvm::zip_equal(lhsDimsVec, rhsDimsVec)) {
    wave::WaveSymbolAttr lhsExpr = lhs.getShape()[lhsDim];
    wave::WaveSymbolAttr rhsExpr = rhs.getShape()[rhsDim];
    if (lhsExpr == rhsExpr)
      continue;

    if (loc) {
      emitError(*loc) << "expected " << lhsName << " dimension #" << lhsDim
                      << " (" << lhsExpr << ") to match " << rhsName
                      << " dimension #" << rhsDim << " (" << rhsExpr << ")";
    }
    return failure();
  }
  return success();
}

llvm::LogicalResult
wave::detail::verifyElementTypesMatch(std::optional<Location> loc,
                                      llvm::StringRef lhsName, Type lhs,
                                      llvm::StringRef rhsName, Type rhs) {
  if (getElementType(lhs) == getElementType(rhs))
    return success();

  if (loc) {
    emitError(*loc) << "expected " << lhsName << " and " << rhsName
                    << " elemental types to match, got " << getElementType(lhs)
                    << ", " << getElementType(rhs);
  }
  return failure();
}

llvm::LogicalResult wave::detail::verifyTensorShapesCompatible(
    wave::WaveTensorType lhs, wave::WaveTensorType rhs,
    std::optional<Location> errorLocation, llvm::StringRef lhsName,
    llvm::StringRef rhsName) {
  if (lhs == rhs)
    return success();

  if (!lhs || !rhs || !lhs.getFullySpecified() || !rhs.getFullySpecified())
    return success();

  if (lhs.getRank() != rhs.getRank()) {
    if (errorLocation) {
      emitError(*errorLocation)
          << "rank mismatch between " << lhsName << " and " << rhsName;
    }
    return failure();
  }

  auto allDims = llvm::to_vector(llvm::iota_range<int>(0, lhs.getRank(),
                                                       /*Inclusive=*/false));
  return verifyTypesMatchingDimensions(errorLocation, lhsName, lhs, allDims,
                                       rhsName, rhs, allDims);
}

llvm::LogicalResult wave::detail::verifyTypesCompatible(
    Type lhs, Type rhs, bool includeAddressSpace, bool includeElementalType,
    std::optional<Location> errorLocation, llvm::StringRef lhsName,
    llvm::StringRef rhsName) {
  // Fast and cheap path.
  if (lhs == rhs)
    return success();

  if (errorLocation) {
    assert(!lhsName.empty() && !rhsName.empty() &&
           "expected names when location is provided");
  }

  if (includeElementalType) {
    if (failed(
            verifyElementTypesMatch(errorLocation, lhsName, lhs, rhsName, rhs)))
      return failure();
  }

  auto lhsTensor = llvm::dyn_cast<wave::WaveTensorType>(lhs);
  auto rhsTensor = llvm::dyn_cast<wave::WaveTensorType>(rhs);
  if (!lhsTensor || !rhsTensor)
    return success();

  if (includeAddressSpace) {
    if (lhsTensor.getAddressSpaceValue() != rhsTensor.getAddressSpaceValue() &&
        lhsTensor.getAddressSpaceValue() !=
            wave::WaveAddressSpace::Unspecified &&
        rhsTensor.getAddressSpaceValue() !=
            wave::WaveAddressSpace::Unspecified) {
      if (errorLocation) {
        emitError(*errorLocation) << "address space mismatch between "
                                  << lhsName << " and " << rhsName;
      }
      return failure();
    }
  }

  return verifyTensorShapesCompatible(lhsTensor, rhsTensor, errorLocation,
                                      lhsName, rhsName);
}

static llvm::LogicalResult
verifyTypeRange(Location loc, TypeRange range, Type referenceType,
                bool includeAddressSpace, bool includeElementalType,
                llvm::StringRef rangeDescriptionPrefix,
                llvm::StringRef referenceDescription) {
  llvm::SmallString<16> rangeDescription(rangeDescriptionPrefix);
  for (auto &&[i, type] : llvm::enumerate(range)) {
    rangeDescription.resize(rangeDescriptionPrefix.size());
    llvm::raw_svector_ostream os(rangeDescription);
    os << i;

    if (failed(wave::detail::verifyTypesCompatible(
            type, referenceType, includeAddressSpace, includeElementalType, loc,
            os.str(), referenceDescription))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyCompatibleOperandsAndResultsOpTrait(
    Operation *op, bool includeAddressSpace, bool includeElementalType) {
  const llvm::StringLiteral kOperandNamePrefix = "operand #";
  const llvm::StringLiteral kResultNamePrefix = "result #";
  std::string referenceDescription;
  llvm::raw_string_ostream os(referenceDescription);
  Type referenceType;
  auto it =
      llvm::find_if(op->getOperandTypes(), llvm::IsaPred<wave::WaveTensorType>);
  auto it2 =
      llvm::find_if(op->getResultTypes(), llvm::IsaPred<wave::WaveTensorType>);
  if (it != op->getOperandTypes().end()) {
    referenceType = *it;
    os << kOperandNamePrefix
       << std::distance(op->getOperandTypes().begin(), it);
  } else if (it2 != op->getResultTypes().end()) {
    referenceType = *it2;
    os << kResultNamePrefix << std::distance(op->getResultTypes().begin(), it2);
  } else if (op->getNumOperands() > 0) {
    referenceType = op->getOperandTypes()[0];
    os << kOperandNamePrefix << 0;
  } else if (op->getNumResults() > 0) {
    referenceType = op->getResultTypes()[0];
    os << kResultNamePrefix << 0;
  } else {
    return llvm::success();
  }

  if (llvm::failed(verifyTypeRange(op->getLoc(), op->getOperandTypes(),
                                   referenceType, includeAddressSpace,
                                   includeElementalType, kOperandNamePrefix,
                                   os.str())))
    return llvm::failure();

  return verifyTypeRange(op->getLoc(), op->getResultTypes(), referenceType,
                         includeAddressSpace, includeElementalType,
                         kResultNamePrefix, os.str());
}

// ----------------------------------------------------------------------------
// Reduction operation traits
// ----------------------------------------------------------------------------

wave::WaveSymbolAttr
wave::detail::getReducedSymbol(Operation *op, wave::WaveSymbolAttr axisAttr,
                               Type inputType) {
  if (axisAttr)
    return axisAttr;

  auto tensor = dyn_cast<wave::WaveTensorType>(inputType);
  if (tensor && tensor.getFullySpecified()) {
    return tensor.getShape().back();
  }
  return {};
}

LogicalResult wave::detail::verifyReductionOperation(Operation *op,
                                                     Type inputTypeBase,
                                                     Type initTypeBase,
                                                     Type resultTypeBase,
                                                     Attribute axisAttr) {
  if (failed(wave::detail::verifyElementTypesMatch(
          op->getLoc(), "input", inputTypeBase, "init", initTypeBase))) {
    return failure();
  }
  if (failed(wave::detail::verifyTypesCompatible(
          initTypeBase, resultTypeBase, /*includeAddressSpace=*/true,
          /*includeElementalType=*/true, op->getLoc(), "init", "result"))) {
    return failure();
  }

  auto inputType = dyn_cast<WaveTensorType>(inputTypeBase);
  auto initType = dyn_cast<WaveTensorType>(initTypeBase);
  auto resultType = dyn_cast<WaveTensorType>(resultTypeBase);

  if (inputType && !inputType.getFullySpecified() && !axisAttr) {
    return op->emitOpError() << "expected axis attribute when input type is "
                             << "not fully specified";
  }

  if (inputType && inputType.getFullySpecified()) {
    if (axisAttr) {
      return op->emitOpError() << "did not expect axis attribute when input "
                                  "type is fully specified";
    }

    if (initType && initType.getFullySpecified()) {
      if (inputType.getRank() - 1 != initType.getRank()) {
        return op->emitOpError()
               << "init tensor rank (" << initType.getRank()
               << ") must be one less than input tensor rank ("
               << inputType.getRank() << ")";
      }
      auto leadingDims = llvm::to_vector(llvm::seq<int>(initType.getRank()));
      if (failed(wave::detail::verifyTypesMatchingDimensions(
              op->getLoc(), "init", initType, leadingDims, "input", inputType,
              leadingDims)))
        return failure();
    }

    if (resultType && resultType.getFullySpecified()) {
      if (inputType.getRank() - 1 != resultType.getRank()) {
        return op->emitOpError()
               << "result tensor rank (" << resultType.getRank()
               << ") must be one less than input tensor rank ("
               << inputType.getRank() << ")";
      }
      auto leadingDims = llvm::to_vector(llvm::seq<int>(resultType.getRank()));
      if (failed(wave::detail::verifyTypesMatchingDimensions(
              op->getLoc(), "input", inputType, leadingDims, "result",
              resultType, leadingDims)))
        return failure();
    }
  }

  if (initType && initType.getFullySpecified()) {
    if (axisAttr && llvm::is_contained(initType.getShape(), axisAttr)) {
      return op->emitOpError()
             << "init tensor shape must not contain the reduced axis";
    }
  }

  if (resultType && resultType.getFullySpecified()) {
    if (axisAttr && llvm::is_contained(resultType.getShape(), axisAttr)) {
      return op->emitOpError()
             << "result tensor shape must not contain the reduced axis";
    }
  }

  return success();
}

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
