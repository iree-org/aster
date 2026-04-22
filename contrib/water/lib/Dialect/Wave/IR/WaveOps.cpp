// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace wave;

//-----------------------------------------------------------------------------
// Custom parsing and printing hooks. These must be defined before including the
// op classes.
//-----------------------------------------------------------------------------

// Parse types of the `wave.register` op and perform type inference. The syntax
// is simply the tensor type from which the elemental type is extract for the
// initializer type.
static ParseResult parseRegisterOpTypes(OpAsmParser &parser, Type &initType,
                                        Type &resultType) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseType(resultType)))
    return failure();

  initType =
      llvm::TypeSwitch<Type, Type>(resultType)
          .Case<wave::WaveTensorType, VectorType>(
              [](auto containerType) { return containerType.getElementType(); })
          .Default([](Type) { return nullptr; });

  if (!initType)
    return parser.emitError(loc)
           << "expected wave tensor or vector type, got " << resultType;

  return success();
}

// Print types of the `wave.register` operation.
static void printRegisterOpTypes(OpAsmPrinter &printer, Operation *,
                                 Type initType, Type resultType) {
#ifndef NDEBUG
  auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(resultType);
  Type elementType = tensorType
                         ? tensorType.getElementType()
                         : llvm::cast<VectorType>(resultType).getElementType();
  assert(initType == elementType && "expected equal types");
#endif // NDEBUG
  (void)initType;
  printer.printType(resultType);
}

// Parse a single type and use it for all operands.
static ParseResult
parseReusedType(OpAsmParser &parser, SmallVectorImpl<Type> &types,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  Type singleType;
  if (failed(parser.parseType(singleType)))
    return failure();

  types.append(operands.size(), singleType);
  return success();
}

// Print the first type assuming all types are equal.
static void printReusedType(OpAsmPrinter &printer, Operation *, TypeRange types,
                            ValueRange) {
  printer.printType(types[0]);
#ifndef NDEBUG
  for (unsigned i = 1, e = types.size(); i < e; ++i) {
    assert(types[i] == types[0] && "expected all types to be equal");
  }
#endif // NDEBUG
}

// Parse an @-symbol and interpret it as a wave symbol.
static ParseResult parseSingleSymbol(OpAsmParser &parser,
                                     wave::WaveSymbolAttr &symbolAttr) {
  StringAttr strAttr;
  if (failed(parser.parseSymbolName(strAttr)))
    return failure();

  symbolAttr =
      wave::WaveSymbolAttr::get(parser.getContext(), strAttr.getValue());
  return success();
}

// Print a wave symbol as an MLIR @-symbol.
static void printSingleSymbol(OpAsmPrinter &printer, Operation *,
                              wave::WaveSymbolAttr symbolAttr) {
  printer.printSymbolName(symbolAttr.getName());
}

#define GET_OP_CLASSES
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"

//-----------------------------------------------------------------------------
// AllocateOp
//-----------------------------------------------------------------------------

llvm::LogicalResult wave::AllocateOp::verify() {
  bool hasParent = getParent() != Value();
  bool hasOffset = getOffset() != std::nullopt;
  if (hasParent ^ hasOffset) {
    return emitOpError()
           << "expects parent and offset to be present simultaneously";
  }

  if (hasParent && getTailPadding())
    return emitOpError() << "only top-level allocations can have tail_padding";

  if (WaveExprListAttr distributedShape = getDistributedShape()) {
    if (distributedShape.getMap().getNumResults() == 0) {
      return emitOpError() << "distributed shape must have at least one result";
    }
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// IterateOp
//-----------------------------------------------------------------------------

void wave::IterateOp::makeIsolated(RewriterBase &rewriter) {
  // Find all uses inside the body of values defined above.
  llvm::SetVector<Value> captures;
  llvm::SmallVector<OpOperand *> captureOperands;
  llvm::SmallPtrSet<Region *, 8> ancestorRegions;
  for (Operation *op = getOperation(); op != nullptr; op = op->getParentOp()) {
    ancestorRegions.insert(op->getParentRegion());
  }
  getOperation()->walk([&](Operation *op) {
    if (op == getOperation())
      return WalkResult::advance();

    for (OpOperand &operand : op->getOpOperands()) {
      if (!ancestorRegions.contains(operand.get().getParentRegion()))
        continue;
      captureOperands.push_back(&operand);
      captures.insert(operand.get());
    }
    return WalkResult::advance();
  });

  // Capture values defined above.
  llvm::SmallVector<Location> newCaptureLocs = llvm::map_to_vector(
      captures, [&](Value value) { return value.getLoc(); });
  rewriter.modifyOpInPlace(
      *this, [&] { getCapturesMutable().append(captures.getArrayRef()); });

  // Add trailing block arguments for captured values. The little dance with the
  // rewriter is a way to append block arguments.
  llvm::SmallVector<Type> allTypes(getLoopBody()->getArgumentTypes());
  llvm::append_range(allTypes, ValueRange(captures.getArrayRef()).getTypes());
  llvm::SmallVector<Location> allLocs =
      llvm::map_to_vector(getLoopBody()->getArguments(),
                          [](Value value) { return value.getLoc(); });
  llvm::append_range(allLocs, newCaptureLocs);
  Block *originalBlock = getLoopBody();
  Block *newBlock = rewriter.createBlock(originalBlock, allTypes, allLocs);
  rewriter.mergeBlocks(originalBlock, newBlock,
                       newBlock->getArguments().drop_back(captures.size()));
  ValueRange innerValues = newBlock->getArguments().take_back(captures.size());

  // Update uses in the body to use block arguments instead of the captured
  // values.
  for (OpOperand *opOperand : captureOperands) {
    rewriter.modifyOpInPlace(opOperand->getOwner(), [&] {
      auto it = llvm::find(captures, opOperand->get());
      assert(it != captures.end() && "expected capture to be found");
      size_t position = std::distance(captures.begin(), it);
      opOperand->set(innerValues[position]);
    });
  }
}

void wave::IterateOp::makeNonIsolated(RewriterBase &rewriter) {
  // Replace uses of block arguments with the captured values, these uses can
  // only be inside the body in well-formed SSA.
  for (auto &&[captureBlockArg, captured] :
       llvm::zip_equal(getCaptureBlockArgs(), getCaptures())) {
    rewriter.replaceAllUsesWith(captureBlockArg, captured);
  }

  // Remove block arguments for captured values that are no longer necessary.
  // The little dance is needed because the rewriter can't directly remove block
  // arguments. Note that it is fine to replace them with nullptr as they have
  // no uses at this point.
  unsigned numCaptures = getCaptures().size();
  rewriter.modifyOpInPlace(*this, [&] { getCapturesMutable().clear(); });
  Block *originalBlock = getLoopBody();
  auto types =
      TypeRange(originalBlock->getArgumentTypes()).drop_back(numCaptures);
  llvm::SmallVector<Location> locations =
      llvm::map_to_vector(originalBlock->getArguments().drop_back(numCaptures),
                          [](Value value) { return value.getLoc(); });
  Block *newBlock = rewriter.createBlock(getLoopBody(), types, locations);
  SmallVector<Value> replacementValues(newBlock->getArguments());
  replacementValues.append(numCaptures, Value());
  rewriter.mergeBlocks(originalBlock, newBlock, replacementValues);
}

bool wave::IterateOp::areTypesCompatible(mlir::Type lhs, mlir::Type rhs) {
  // Handle both WaveTensorType and VectorType combinations.
  auto lhsTensor = llvm::dyn_cast<wave::WaveTensorType>(lhs);
  auto rhsTensor = llvm::dyn_cast<wave::WaveTensorType>(rhs);
  auto lhsVector = llvm::dyn_cast<mlir::VectorType>(lhs);
  auto rhsVector = llvm::dyn_cast<mlir::VectorType>(rhs);

  // Both are wave tensors - check shape and address space compatibility.
  if (lhsTensor && rhsTensor) {
    return detail::verifyTypesCompatible(lhsTensor, rhsTensor,
                                         /*includeAddressSpace=*/true,
                                         /*includeElementalType=*/true)
        .succeeded();
  }

  // Both are vectors - simple equality check.
  if (lhsVector && rhsVector) {
    return lhsVector == rhsVector;
  }

  // Mixed types are not compatible.
  return false;
}

OperandRange wave::IterateOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getOperands().drop_back(getNumOperands());
}

void wave::IterateOp::getSuccessorRegions(
    RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::RegionSuccessor> &regions) {
  // May branch into the region or bypass it regardless of the source.
  regions.emplace_back(RegionSuccessor::parent());
  regions.emplace_back(RegionSuccessor(&getBody()));
}

LogicalResult wave::IterateOp::verify() {
  if (getNumOperands() != getLoopBody()->getNumArguments()) {
    return emitOpError() << "expects the same number of operands ("
                         << getNumOperands() << ") and block arguments ("
                         << getLoopBody()->getNumArguments() << ")";
  }
  TypeRange blockIterArgTypes = getIterArgs().getTypes();
  TypeRange iterArgTypes =
      getOperands().drop_back(getCaptures().size()).getTypes();
  TypeRange resultTypes = getResultTypes();
  if (iterArgTypes.size() != blockIterArgTypes.size()) {
    return emitOpError() << "expects the same number if iter_args ("
                         << iterArgTypes.size()
                         << ") and corresponding block arguments ("
                         << blockIterArgTypes.size() << ")";
  }
  if (blockIterArgTypes.size() != resultTypes.size()) {
    return emitOpError() << "expects the same number of iter_args ("
                         << blockIterArgTypes.size() << ") and results ("
                         << resultTypes.size() << ")";
  }
  for (auto &&[i, iterArg, result] :
       llvm::enumerate(iterArgTypes, resultTypes)) {
    // Handle verification for both wave tensors and vectors.
    auto iterArgTensor = llvm::dyn_cast<wave::WaveTensorType>(iterArg);
    auto resultTensor = llvm::dyn_cast<wave::WaveTensorType>(result);

    auto istr = std::to_string(i);

    // Both are wave tensors - verify shapes match across all dimensions.
    if (iterArgTensor && resultTensor) {
      if (!iterArgTensor.getFullySpecified() ||
          !resultTensor.getFullySpecified())
        continue;

      auto allDims =
          llvm::to_vector(llvm::iota_range<int>(0, iterArgTensor.getRank(),
                                                /*Inclusive=*/false));
      if (mlir::failed(detail::verifyTypesMatchingDimensions(
              getLoc(), "iter_args #" + istr, iterArgTensor, allDims,
              "result #" + istr, resultTensor, allDims)))
        return mlir::failure();
    }
    // Both are vectors - check exact type equality.
    else if (isa<VectorType>(iterArg) && isa<VectorType>(result)) {
      if (iterArg != result) {
        return emitOpError()
               << "iter_args #" << i << " type (" << iterArg
               << ") must match result #" << i << " type (" << result << ")";
      }
    }
    // Mixed types are not allowed.
    else {
      return emitOpError() << "iter_args #" << i << " and result #" << i
                           << " must be the same category of types (both wave "
                              "tensors or both vectors)";
    }
  }

  return llvm::success();
}

llvm::LogicalResult wave::IterateOp::verifyRegions() {
  // Use the region hook since it runs after we verified the terminator itself
  // and know it is well-formed.
  TypeRange iterArgTypes = getIterArgs().getTypes();
  TypeRange blockIterArgTypes = TypeRange(getLoopBody()->getArgumentTypes())
                                    .take_front(iterArgTypes.size());
  TypeRange resultTypes = getResultTypes();
  TypeRange terminatorOperandTypes =
      getLoopBody()->getTerminator()->getOperands().getTypes();
  if (resultTypes.size() != terminatorOperandTypes.size()) {
    return emitOpError() << "expects the same number of results ("
                         << resultTypes.size() << ") and terminator operands ("
                         << terminatorOperandTypes.size() << ")";
  }
  for (auto &&[i, result, terminatorOperand, iterArg, blockIterArg] :
       llvm::enumerate(resultTypes, terminatorOperandTypes, iterArgTypes,
                       blockIterArgTypes)) {
    auto istr = std::to_string(i);

    auto iterArgTensor = llvm::dyn_cast<wave::WaveTensorType>(iterArg);
    auto resultTensor = llvm::dyn_cast<wave::WaveTensorType>(result);
    auto blockIterArgTensor =
        llvm::dyn_cast<wave::WaveTensorType>(blockIterArg);
    auto terminatorOperandTensor =
        llvm::dyn_cast<wave::WaveTensorType>(terminatorOperand);

    // Verify result type vs terminator operand type.
    if (resultTensor && terminatorOperandTensor) {
      if (llvm::failed(detail::verifyTypesCompatible(
              resultTensor, terminatorOperandTensor,
              /*includeAddressSpace=*/true, /*includeElementalType=*/true,
              getLoc(), "result #" + istr, "terminator operand #" + istr))) {
        return llvm::failure();
      }
    } else if (isa<VectorType>(result) && isa<VectorType>(terminatorOperand)) {
      // For vector types, just check that they are exactly equal.
      if (result != terminatorOperand) {
        return emitOpError() << "result #" << i << " type (" << result
                             << ") does not match terminator operand #" << i
                             << " type (" << terminatorOperand << ")";
      }
    } else if (result != terminatorOperand) {
      return emitOpError() << "result #" << i << " type (" << result
                           << ") and terminator operand #" << i << " type ("
                           << terminatorOperand << ") are not compatible types";
    }

    // Verify iter arg type vs block arg type.
    if (iterArgTensor && blockIterArgTensor) {
      if (llvm::failed(detail::verifyTypesCompatible(
              iterArgTensor, blockIterArgTensor,
              /*includeAddressSpace=*/true, /*includeElementalType=*/true,
              getLoc(), "iter arg #" + istr, "block iter arg #" + istr))) {
        return llvm::failure();
      }
    } else if (isa<VectorType>(iterArg) && isa<VectorType>(blockIterArg)) {
      // For vector types, just check that they are exactly equal.
      if (iterArg != blockIterArg) {
        return emitOpError() << "iter arg #" << i << " type (" << iterArg
                             << ") does not match block iter arg #" << i
                             << " type (" << blockIterArg << ")";
      }
    } else if (iterArg != blockIterArg) {
      return emitOpError() << "iter arg #" << i << " type (" << iterArg
                           << ") and block iter arg #" << i << " type ("
                           << blockIterArg << ") are not compatible types";
    }
  }

  // Verify capture types match their corresponding block arguments.
  TypeRange captureTypes = getCaptures().getTypes();
  TypeRange captureBlockArgTypes = TypeRange(getLoopBody()->getArgumentTypes())
                                       .take_back(captureTypes.size());
  for (auto &&[i, capture, captureBlockArg] :
       llvm::enumerate(captureTypes, captureBlockArgTypes)) {
    if (capture != captureBlockArg) {
      return emitOpError() << "expects the same type for capture #" << i
                           << " and block argument #"
                           << (getIterArgs().size() + i);
    }
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// MmaOp
//-----------------------------------------------------------------------------

llvm::FailureOr<ChangeResult> wave::MmaOp::propagateForward(
    llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  // TODO: probably upstream, we want a value-less adaptor or some way of
  // querying positions of named arguments instead of magic values here. It is
  // currently possible by doing get<OperandName>Mutable().getOperandNumber(),
  // but a bit annoying.
  return detail::propagateShapeInformation(operandTypes[2], resultTypes[0],
                                           "accumulator", "result", errs);
}

llvm::FailureOr<ChangeResult> wave::MmaOp::propagateBackward(
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  // TODO: we may consider doing partial type propagation, but we can't infer
  // the reduction dimension from the result type since it is absent from it.
  return detail::propagateShapeInformation(resultTypes[0], operandTypes[2],
                                           "result", "accumulator", errs);
}

LogicalResult wave::MmaOp::finalizeTypeInference() { return success(); }

// Check if the given type is one of the allowed types provided as template
// arguments and report an error at the given location otherwise.
template <typename... AllowedTypes>
static LogicalResult checkAllowedTypes(Location loc, Type type,
                                       llvm::StringRef name,
                                       wave::WaveMmaKind kind) {
  if (llvm::isa<AllowedTypes...>(type))
    return success();

  return emitError(loc) << "unexpected " << name << " elemental type " << type
                        << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the given type is an integer type with one of the provided bitwidths
// and report and error at the given location otherwise.
template <typename T>
static std::enable_if_t<std::is_same_v<T, IntegerType>, LogicalResult>
checkAllowedTypes(Location loc, Type type, llvm::StringRef name,
                  wave::WaveMmaKind kind, llvm::ArrayRef<unsigned> bitwidths) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    if (llvm::is_contained(bitwidths, intType.getIntOrFloatBitWidth()))
      return success();
  }
  return emitError(loc) << "unexpected " << name << " elemental type " << type
                        << " for MMA kind " << wave::stringifyEnum(kind);
}

// Check if the types used for multiplication and accumulation in a `wave.mma`
// operation are compatible with the specified MMA kind and report an error
// otherwise.
static LogicalResult checkMmaTypeCompatibility(Location loc,
                                               wave::WaveMmaKind kind,
                                               Type mulType, Type accType) {
  bool success = false;
  switch (kind) {
  case wave::WaveMmaKind::F32_16x16x16_F16:
  case wave::WaveMmaKind::F32_32x32x8_F16:
  case wave::WaveMmaKind::F32_16x16x32_K8_F16:
  case wave::WaveMmaKind::F32_32x32x16_F16:
  case wave::WaveMmaKind::F32_16x16x32_F16:
  case wave::WaveMmaKind::F32_32x32x16_K8_F16:
  case wave::WaveMmaKind::F32_32x32x16_BF16:
  case wave::WaveMmaKind::F32_16x16x32_BF16: {
    success = succeeded(checkAllowedTypes<Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              succeeded(checkAllowedTypes<Float16Type, BFloat16Type>(
                  loc, mulType, "lhs/rhs", kind));
    break;
  }

  case wave::WaveMmaKind::I32_16x16x16_I8:
  case wave::WaveMmaKind::I32_32x32x8_I8:
  case wave::WaveMmaKind::I32_16x16x32_I8:
  case wave::WaveMmaKind::I32_32x32x16_I8:
    success = succeeded(checkAllowedTypes<IntegerType>(
                  loc, accType, "accumulator/result", kind, {32})) &&
              succeeded(checkAllowedTypes<IntegerType>(loc, mulType, "lhs/rhs",
                                                       kind, {8}));
    break;

  case wave::WaveMmaKind::F32_16x16x32_F8:
  case wave::WaveMmaKind::F32_32x32x16_F8:
  case wave::WaveMmaKind::F32_16x16x32_K4_F8:
  case wave::WaveMmaKind::F32_32x32x16_K4_F8:
    success = succeeded(checkAllowedTypes<Float32Type>(
                  loc, accType, "accumulator/result", kind)) &&
              succeeded(checkAllowedTypes<Float8E3M4Type, Float8E5M2Type>(
                  loc, mulType, "lhs/rhs", kind));
    break;

  case wave::WaveMmaKind::F32_16x16x128_F8F6F4:
  case wave::WaveMmaKind::F32_32x32x64_F8F6F4:
    success =
        succeeded(checkAllowedTypes<Float32Type>(loc, accType,
                                                 "accumulator/result", kind)) &&
        succeeded(
            checkAllowedTypes<Float8E3M4Type, Float8E5M2Type, Float6E2M3FNType,
                              Float6E3M2FNType, Float4E2M1FNType>(
                loc, mulType, "lhs/rhs", kind));
    break;
  }

  return llvm::success(success);
}

LogicalResult MmaOp::verify() {
  Type lhsTypeGeneric = getLhs().getType();
  Type rhsTypeGeneric = getRhs().getType();
  Type accumulatorTypeGeneric = getAccumulator().getType();
  Type resultTypeGeneric = getResult().getType();

  WaveTensorType lhsType = dyn_cast<wave::WaveTensorType>(lhsTypeGeneric);
  WaveTensorType rhsType = dyn_cast<wave::WaveTensorType>(rhsTypeGeneric);
  WaveTensorType accumulatorType =
      dyn_cast<wave::WaveTensorType>(accumulatorTypeGeneric);
  WaveTensorType resultType = dyn_cast<wave::WaveTensorType>(resultTypeGeneric);

  // TODO: need to verify vector types, but for that, we need to know what they
  // must look like based on the MMA enum.
  if (!lhsType || !rhsType || !accumulatorType || !resultType) {
    return success();
  }

  if (failed(detail::verifyElementTypesMatch(getLoc(), "LHS", lhsType, "RHS",
                                             rhsType)) ||
      failed(detail::verifyElementTypesMatch(getLoc(), "result", resultType,
                                             "accumulator", accumulatorType)))
    return failure();

  if (lhsType.getRank() != rhsType.getRank() ||
      lhsType.getRank() != accumulatorType.getRank() ||
      lhsType.getRank() != resultType.getRank()) {
    return emitOpError()
           << "expects all operands and results to have the same rank";
  }

  if (lhsType.getRank() < 2)
    return emitOpError() << "expects at least 2D operands for MMA";

  SmallVector<int> batchDims =
      llvm::to_vector(llvm::seq<int>(0, lhsType.getRank() - 2));
  SmallVector<int> batchAndLast(batchDims);
  batchAndLast.push_back(lhsType.getRank() - 1);
  SmallVector<int> batchAndSecondToLast(std::move(batchDims));
  batchAndSecondToLast.push_back(lhsType.getRank() - 2);

  if (detail::verifyTypesMatchingDimensions(
          getLoc(), "LHS", lhsType, batchAndLast, "RHS", rhsType, batchAndLast)
          .failed() ||
      detail::verifyTypesMatchingDimensions(
          getLoc(), "LHS", lhsType, batchAndSecondToLast, "accumulator",
          accumulatorType, batchAndSecondToLast)
          .failed() ||
      detail::verifyTypesMatchingDimensions(getLoc(), "RHS", rhsType,
                                            batchAndSecondToLast, "accumulator",
                                            accumulatorType, batchAndLast)
          .failed()) {
    return failure();
  }

  if (!getKind())
    return success();

  return checkMmaTypeCompatibility(getLoc(), *getKind(),
                                   lhsType.getElementType(),
                                   accumulatorType.getElementType());
}

//-----------------------------------------------------------------------------
// ReadOp
//-----------------------------------------------------------------------------

// Propagate the shape information from `from` to `to`.
static FailureOr<ChangeResult>
propagateTypes(wave::WaveTensorType from, wave::WaveTensorType &to,
               StringRef fromName, StringRef toName, raw_ostream &errs) {
  if (!from) {
    if (!to)
      return ChangeResult::NoChange;
    to = nullptr;
    return ChangeResult::Change;
  }
  if (!from.getFullySpecified())
    return ChangeResult::NoChange;

  return wave::detail::propagateShapeInformation(from, to, fromName, toName,
                                                 errs);
}

FailureOr<ChangeResult>
ReadOp::propagateForward(ArrayRef<wave::WaveTensorType> operandTypes,
                         MutableArrayRef<wave::WaveTensorType> resultTypes,
                         raw_ostream &errs) {
  return propagateTypes(operandTypes[0], resultTypes[0], "memory", "value",
                        errs);
}

FailureOr<ChangeResult>
ReadOp::propagateBackward(MutableArrayRef<wave::WaveTensorType> operandTypes,
                          ArrayRef<wave::WaveTensorType> resultTypes,
                          raw_ostream &errs) {
  return propagateTypes(resultTypes[0], operandTypes[0], "value", "memory",
                        errs);
}

LogicalResult ReadOp::finalizeTypeInference() { return success(); }

// Check the correspondence of the explicit elements per thread, if provided,
// with the number of elements in the vector type.
static LogicalResult
verifyIndexElementsPerThread(Operation *op,
                             std::optional<int64_t> elementsPerThread,
                             Type maybeVectorType) {
  auto vectorType = dyn_cast<VectorType>(maybeVectorType);
  auto vectorSize = vectorType
                        ? std::optional<int64_t>(vectorType.getDimSize(0))
                        : std::nullopt;

  if (elementsPerThread && vectorSize && *elementsPerThread != *vectorSize) {
    return op->emitOpError()
           << "expected result vector type to have the "
              "number of elements per thread matching the attribute ("
           << *elementsPerThread << "), got " << vectorType.getDimSize(0);
  }

  return success();
}

/// Common verification logic for ReadOp and WriteOp.
static LogicalResult verifyReadWriteOp(Operation *op,
                                       std::optional<int64_t> elementsPerThread,
                                       Type memoryType, Type valueType) {

  if (failed(wave::detail::verifyElementTypesMatch(
          op->getLoc(), "memory", memoryType, "register", valueType)))
    return failure();

  auto memoryTensorType = dyn_cast<WaveTensorType>(memoryType);
  auto valueTensorType = dyn_cast<WaveTensorType>(valueType);

  if (memoryTensorType && valueTensorType &&
      memoryTensorType.getFullySpecified() &&
      valueTensorType.getFullySpecified()) {
    if (!llvm::equal(valueTensorType.getShape(), memoryTensorType.getShape())) {
      InFlightDiagnostic diag = op->emitError() << "the value shape (";
      llvm::interleaveComma(
          valueTensorType.getShape(), diag,
          [&](WaveSymbolAttr symbol) { diag << symbol.getName(); });
      diag << ") doesn't match the memory shape (";
      llvm::interleaveComma(
          memoryTensorType.getShape(), diag,
          [&](WaveSymbolAttr symbol) { diag << symbol.getName(); });
      diag << ")";
      return diag;
    }
  }

  if (!memoryTensorType)
    return success();

  return verifyIndexElementsPerThread(op, elementsPerThread, valueType);
}

LogicalResult ReadOp::verify() {
  return verifyReadWriteOp(*this, getElementsPerThread(), getMemory().getType(),
                           getResult().getType());
}

//-----------------------------------------------------------------------------
// RegisterOp
//-----------------------------------------------------------------------------

LogicalResult wave::RegisterOp::verify() {
  Type type = getResult().getType();
  auto tensorType = dyn_cast<WaveTensorType>(type);
  auto elementType = tensorType ? tensorType.getElementType()
                                : cast<VectorType>(type).getElementType();
  Type initType = getInit().getType();
  if (elementType != initType) {
    return emitOpError() << "expected the type of the init value to match the "
                            "elemental type of the result";
  }
  if (!tensorType)
    return success();

  if (!tensorType.getFullySpecified()) {
    return emitOpError() << "expected fully-specified tensor type";
  }
  return success();
}

//-----------------------------------------------------------------------------
// ExtractOp
//-----------------------------------------------------------------------------

LogicalResult ExtractOp::verify() {
  wave::WaveExprListAttr position = getPosition();
  if (position.getRank() != 1) {
    return emitOpError() << "position must contain exactly one expression, but "
                            "got "
                         << position.getRank();
  }

  if (failed(detail::verifyElementTypesMatch(getLoc(), "source",
                                             getSource().getType(), "result",
                                             getResult().getType()))) {
    return failure();
  }

  if (auto resultVectorType = dyn_cast<VectorType>(getResult().getType())) {
    if (resultVectorType.getShape()[0] != 1) {
      return emitOpError() << "result must be a 1-element vector, got "
                           << resultVectorType;
    }
    return success();
  }

  auto sourceTensorType = dyn_cast<WaveTensorType>(getSource().getType());
  // For mixed types, cannot do anything here.
  if (!sourceTensorType)
    return success();

  if (!sourceTensorType.getFullySpecified())
    return emitOpError() << "source tensor type must be fully specified";

  auto resultTensorType = cast<WaveTensorType>(getResult().getType());

  if (!resultTensorType.getFullySpecified())
    return emitOpError() << "target tensor type must be fully specified";

  if (resultTensorType.getRank() + 1 != sourceTensorType.getRank()) {
    return emitOpError()
           << "result tensor must have one less dimension than source";
  }

  for (WaveSymbolAttr dim : resultTensorType.getShape()) {
    if (!llvm::is_contained(sourceTensorType.getShape(), dim)) {
      return emitOpError() << "source tensor type dimensions must contain the "
                              "result tensor type dimension "
                           << dim;
    }
  }

  return success();
}

//-----------------------------------------------------------------------------
// ExtractSliceOp
//-----------------------------------------------------------------------------

LogicalResult ExtractSliceOp::verify() {
  wave::WaveExprListAttr offset = getOffset();
  wave::WaveExprListAttr size = getSize();
  wave::WaveExprListAttr stride = getStride();

  if (failed(wave::verifyExprAttrsSameRank({offset, size, stride}))) {
    return emitOpError() << "offset, size, and stride must all have the same "
                            "rank, but got offset rank "
                         << offset.getRank() << ", size rank " << size.getRank()
                         << ", and stride rank " << stride.getRank();
  }

  if (failed(wave::verifyExprAttrsNoSymbols({offset, size, stride}))) {
    return emitOpError() << "offset, size, and stride must be constant "
                            "expressions with no symbols, but got offset with "
                         << offset.getNumSymbols() << " symbols, size with "
                         << size.getNumSymbols() << " symbols, and stride with "
                         << stride.getNumSymbols() << " symbols";
  }

  return success();
}

//-----------------------------------------------------------------------------
// ReshapeOp
//-----------------------------------------------------------------------------

LogicalResult ReshapeOp::verify() {
  if (getSource().empty()) {
    return emitOpError() << "expected at least one source operand";
  }

  Type sourceType = getSource().front().getType();
  for (unsigned i = 1, e = getSource().size(); i < e; ++i) {
    Type currentSourceType = getSource()[i].getType();
    if (currentSourceType != sourceType) {
      return emitOpError()
             << "expected all source operands to have the same type";
    }
  }

  if (failed(detail::verifyElementTypesMatch(
          getLoc(), "source", sourceType, "result", getResult().getType()))) {
    return failure();
  }

  if (getLogicalSlice() >= getNumSlices()) {
    return emitOpError()
           << "expected logical slice to be less than the number of slices";
  }

  // We already verified that source types are equal vector types if there's
  // more than one.
  auto sourceVecType = llvm::dyn_cast<VectorType>(sourceType);
  auto resultVecType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (resultVecType && sourceVecType) {
    int64_t resultNumElems = resultVecType.getNumElements();
    int64_t operandNumElems = sourceVecType.getNumElements();
    if (getSource().size() > 1) {
      unsigned numOperands = getSource().size();

      if (!(operandNumElems == resultNumElems ||
            (operandNumElems * numOperands == resultNumElems))) {
        return emitOpError() << "the total number of elements must remain the "
                                "same or be a concatenation";
      }
    } else {
      if (static_cast<uint64_t>(operandNumElems) !=
          resultNumElems * getNumSlices()) {
        return emitOpError() << "expects operand vector to have "
                             << resultNumElems * getNumSlices()
                             << " elements, got " << operandNumElems;
      }
    }
  }

  auto sourceTensorType = dyn_cast<WaveTensorType>(sourceType);
  auto resultTensorType = dyn_cast<WaveTensorType>(getResult().getType());
  if (!sourceTensorType || !resultTensorType ||
      !sourceTensorType.getFullySpecified() ||
      !resultTensorType.getFullySpecified()) {
    return success();
  }

  if (!getTargetVectorShape().contains(
          resultTensorType.getShape().back().getName())) {
    return emitOpError() << "target_vector_shape must contain at least the "
                            "last dimension of the result tensor type";
  }
  for (auto symbol : getTargetVectorShape()) {
    if (llvm::none_of(resultTensorType.getShape(), [symbol](WaveSymbolAttr s) {
          return s.getName() == symbol.getName();
        })) {
      return emitOpError() << "target_vector_shape contains symbol "
                           << symbol.getName().strref()
                           << " that is not present in the result tensor type";
    }
  }
  return success();
}

//-----------------------------------------------------------------------------
// WriteOp
//-----------------------------------------------------------------------------

LogicalResult WriteOp::verify() {
  return verifyReadWriteOp(*this, getElementsPerThread(), getMemory().getType(),
                           getValueToStore().getType());
}

FailureOr<ChangeResult>
WriteOp::propagateForward(ArrayRef<wave::WaveTensorType>,
                          MutableArrayRef<wave::WaveTensorType>,
                          raw_ostream &) {
  // WriteOp has no results; forward propagation only updates result types.
  return ChangeResult::NoChange;
}

FailureOr<ChangeResult>
WriteOp::propagateBackward(MutableArrayRef<wave::WaveTensorType> operandTypes,
                           ArrayRef<wave::WaveTensorType> resultTypes,
                           raw_ostream &errs) {
  return propagateTypes(operandTypes[1], operandTypes[0], "memory", "value",
                        errs) |
         propagateTypes(operandTypes[0], operandTypes[1], "value", "memory",
                        errs);
}

LogicalResult WriteOp::finalizeTypeInference() { return success(); }

//-----------------------------------------------------------------------------
// YieldOp
//-----------------------------------------------------------------------------

MutableOperandRange
wave::YieldOp::getMutableSuccessorOperands(RegionSuccessor) {
  // Create an empty mutable operand range (it has no default constructor).
  return getValuesMutable().slice(/*subStart=*/0, /*subLen=*/0);
}

//-----------------------------------------------------------------------------
// CastOp
//-----------------------------------------------------------------------------

LogicalResult wave::CastOp::verify() {
  Type valueType = getValueToCast().getType();
  Type resultType = getResult().getType();

  wave::WaveTensorType valueTensor =
      llvm::dyn_cast<wave::WaveTensorType>(valueType);
  wave::WaveTensorType resultTensor =
      llvm::dyn_cast<wave::WaveTensorType>(resultType);
  VectorType valueVec = llvm::dyn_cast<VectorType>(valueType);
  VectorType resultVec = llvm::dyn_cast<VectorType>(resultType);
  if (valueTensor && resultTensor && valueTensor.getFullySpecified() &&
      resultTensor.getFullySpecified() &&
      valueTensor.getShape() != resultTensor.getShape()) {
    return emitOpError("shape of input (")
           << valueTensor.getShape() << ") must match shape of result ("
           << resultTensor.getShape() << ")";
  }

  if (valueVec && resultVec && valueVec.getShape() != resultVec.getShape()) {
    return emitOpError("shape of input (")
           << valueVec.getShape() << ") must match shape of result ("
           << resultVec.getShape() << ")";
  }

  return success();
}

//-----------------------------------------------------------------------------
// ReciprocalOp
//-----------------------------------------------------------------------------

LogicalResult wave::ReciprocalOp::verify() {
  Type argType = getArgument().getType();
  Type elementType = wave::getElementType(argType);
  if (!isa<FloatType>(elementType))
    return emitOpError("requires float element type, but got ") << elementType;

  return success();
}

//-----------------------------------------------------------------------------
// SelectOp
//-----------------------------------------------------------------------------

LogicalResult wave::SelectOp::verify() {
  if (failed(detail::verifyTypesCompatible(
          getLhs().getType(), getRhs().getType(),
          /*includeAddressSpace=*/false, /*includeElementalType=*/true,
          getLoc(), "LHS", "RHS")))
    return failure();

  if (failed(detail::verifyTypesCompatible(
          getLhs().getType(), getResult().getType(),
          /*includeAddressSpace=*/false, /*includeElementalType=*/true,
          getLoc(), "LHS", "result")))
    return failure();

  auto intType =
      dyn_cast<IntegerType>(getElementType(getCondition().getType()));
  if (!intType || intType.getWidth() != 1)
    return emitOpError("condition must be a tensor or vector of i1");

  SmallVector<int64_t> vecShapes;
  if (auto conditionVec = dyn_cast<VectorType>(getCondition().getType()))
    vecShapes.push_back(conditionVec.getNumElements());
  if (auto lhsVec = dyn_cast<VectorType>(getLhs().getType()))
    vecShapes.push_back(lhsVec.getNumElements());
  if (auto rhsVec = dyn_cast<VectorType>(getRhs().getType()))
    vecShapes.push_back(rhsVec.getNumElements());
  if (auto resultVec = dyn_cast<VectorType>(getResult().getType()))
    vecShapes.push_back(resultVec.getNumElements());

  for (int64_t i = 0, e = vecShapes.size(); i < e; ++i) {
    for (int64_t j = i + 1; j < e; ++j) {
      if (vecShapes[i] == vecShapes[j])
        continue;

      emitError() << "expects all vector shapes to be equal, got "
                  << vecShapes[i] << " and " << vecShapes[j];
    }
  }

  auto conditionTensor = dyn_cast<WaveTensorType>(getCondition().getType());
  auto lhsTensor = dyn_cast<WaveTensorType>(getLhs().getType());
  auto rhsTensor = dyn_cast<WaveTensorType>(getRhs().getType());
  auto resultTensor = dyn_cast<WaveTensorType>(getResult().getType());
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, lhsTensor, getLoc(), "condition", "LHS")))
    return failure();
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, rhsTensor, getLoc(), "condition", "RHS")))
    return failure();
  if (failed(detail::verifyTensorShapesCompatible(
          conditionTensor, resultTensor, getLoc(), "condition", "result")))
    return failure();

  return success();
}

//-----------------------------------------------------------------------------
// SelfIndexOp
//-----------------------------------------------------------------------------

LogicalResult wave::SelfIndexOp::verify() {
  Type elementType = getElementType(getResult().getType());

  if (!isa<IntegerType>(elementType))
    return emitOpError() << "result element type must be an integer type, got "
                         << elementType;

  auto tensorType = dyn_cast<WaveTensorType>(getResult().getType());
  if (!tensorType)
    return success();

  if (!tensorType.getFullySpecified())
    return success();

  if (tensorType.getRank() != 1)
    return emitOpError() << "result must be a 1-dimensional tensor, got rank "
                         << tensorType.getRank();

  if (tensorType.getShape()[0] != getDim())
    return emitOpError() << "result dimension '"
                         << tensorType.getShape()[0].getName()
                         << "' must match the specified dimension '"
                         << getDim().getName() << "'";

  return verifyIndexElementsPerThread(getOperation(), getElementsPerThread(),
                                      getResult().getType());
}

//-----------------------------------------------------------------------------
// BroadcastOp
//-----------------------------------------------------------------------------

llvm::SmallVector<WaveSymbolAttr> wave::BroadcastOp::inferBroadcastDims() {
  WaveTensorType sourceType = llvm::cast<WaveTensorType>(getSource().getType());
  WaveTensorType resultType = llvm::cast<WaveTensorType>(getResult().getType());
  assert(sourceType.getFullySpecified() && resultType.getFullySpecified() &&
         "expected source and result types to be fully specified");

  llvm::DenseSet<WaveSymbolAttr> sourceSymbols;
  for (WaveSymbolAttr sym : sourceType.getShape())
    sourceSymbols.insert(sym);

  llvm::SmallVector<WaveSymbolAttr> broadcastDims;
  for (WaveSymbolAttr sym : resultType.getShape()) {
    if (!sourceSymbols.contains(sym))
      broadcastDims.push_back(sym);
  }
  return broadcastDims;
}

LogicalResult wave::BroadcastOp::verify() {
  if (failed(detail::verifyElementTypesMatch(getLoc(), "source",
                                             getSource().getType(), "result",
                                             getResult().getType())))
    return failure();

  auto sourceType = llvm::dyn_cast<WaveTensorType>(getSource().getType());
  auto resultType = llvm::dyn_cast<WaveTensorType>(getResult().getType());

  if (!sourceType || !resultType)
    return success();

  // When result is a tensor, require it to be fully specified (vectors are
  // unchanged).
  if (!resultType.getFullySpecified())
    return emitOpError(
        "result type must be fully specified when it is a tensor");

  // Check all source symbols are in result and in the correct order.
  ArrayRef<WaveSymbolAttr> remainingResultShape = resultType.getShape();
  for (WaveSymbolAttr sym : sourceType.getShape()) {
    auto it = llvm::find(remainingResultShape, sym);
    if (it == remainingResultShape.end()) {
      if (llvm::is_contained(resultType.getShape(), sym)) {
        return emitOpError() << "source dimension " << sym.getName()
                             << " is reordered with respect to other source "
                                "dimensions in the result shape";
      }
      return emitOpError("source dimension '")
             << sym.getName() << "' not found in result shape";
    }
    remainingResultShape = remainingResultShape.drop_front(
        std::distance(remainingResultShape.begin(), it) + 1);
  }

  return success();
}

//-----------------------------------------------------------------------------
// PermuteOp
//-----------------------------------------------------------------------------

/// Helper to validate the input type of a permute operation.
/// Checks if the input shape is a permutation of the result shape.
static LogicalResult validatePermutationInput(WaveTensorType inputType,
                                              WaveTensorType resultType,
                                              llvm::raw_ostream &errs) {
  // We cannot validate unspecified types.
  if (!inputType.getFullySpecified() || !resultType.getFullySpecified())
    return llvm::success();

  if (inputType.getShape().size() != resultType.getShape().size()) {
    errs << "input shape rank (" << inputType.getShape().size()
         << ") does not match target shape rank ("
         << resultType.getShape().size() << ")";
    return failure();
  }

  llvm::SmallDenseSet<WaveSymbolAttr, 4> resultShapeSet;
  resultShapeSet.insert_range(resultType.getShape());

  for (auto inputDim : inputType.getShape()) {
    if (!resultShapeSet.contains(inputDim)) {
      errs << "input dimension '" << inputDim.getName()
           << "' is not present in result shape";
      return failure();
    }
  }

  return llvm::success();
}

LogicalResult wave::PermuteOp::verify() {
  Value input = getValue();
  Value result = getResult();

  if (failed(detail::verifyElementTypesMatch(getLoc(), "input", input.getType(),
                                             "result", result.getType())))
    return failure();

  auto inputType = dyn_cast<WaveTensorType>(input.getType());
  auto resultType = dyn_cast<WaveTensorType>(result.getType());

  // If result / input is a vector (post-lowering phase), skip wave tensor
  // checks.
  if (!resultType || !inputType)
    return success();

  if (!inputType.getFullySpecified() || !resultType.getFullySpecified())
    return success();

  std::string errorMessage;
  llvm::raw_string_ostream errs(errorMessage);
  if (failed(validatePermutationInput(inputType, resultType, errs))) {
    return emitOpError() << errorMessage;
  }

  return success();
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateForward(
    llvm::ArrayRef<wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  unsigned inputOperandPosition = getValueMutable().getOperandNumber();
  WaveTensorType inputType = operandTypes[inputOperandPosition];
  WaveTensorType &resultType = resultTypes[0];

  // Skip validation if either type is not fully specified.
  if (!inputType || !inputType.getFullySpecified() || !resultType ||
      !resultType.getFullySpecified())
    return ChangeResult::NoChange;

  if (failed(validatePermutationInput(inputType, resultType, errs)))
    return llvm::failure();

  return ChangeResult::NoChange;
}

llvm::FailureOr<ChangeResult> wave::PermuteOp::propagateBackward(
    llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<wave::WaveTensorType> resultTypes, llvm::raw_ostream &errs) {
  unsigned inputOperandPosition = getValueMutable().getOperandNumber();
  WaveTensorType inputType = operandTypes[inputOperandPosition];
  WaveTensorType resultType = resultTypes[0];

  if (!resultType || !resultType.getFullySpecified() || !inputType ||
      !inputType.getFullySpecified())
    return ChangeResult::NoChange;

  if (failed(validatePermutationInput(inputType, resultType, errs)))
    return llvm::failure();

  // Cannot propagate shape information backward for permute operations
  // because the input shape ordering is not determined by the result.
  return ChangeResult::NoChange;
}

llvm::LogicalResult wave::PermuteOp::finalizeTypeInference() {
  return llvm::success();
}
