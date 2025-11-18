//===- Utils.cpp ----------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::aster;

//===----------------------------------------------------------------------===//
// FuncTypeConverter
//===----------------------------------------------------------------------===//

FunctionType
FuncTypeConverter::convertFunctionSignatureImpl(const TypeConverter &converter,
                                                FunctionType funcTy,
                                                SignatureConversion &result) {
  SmallVector<Type, 8> ins, outs;
  for (auto [idx, type] : llvm::enumerate(funcTy.getInputs())) {
    ins.clear();
    if (failed(converter.convertTypes(type, ins)))
      return {};
    result.addInputs(idx, ins);
  }
  if (failed(converter.convertTypes(funcTy.getResults(), outs)))
    return {};
  return FunctionType::get(funcTy.getContext(), result.getConvertedTypes(),
                           outs);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
//===----------------------------------------------------------------------===//
// CallOpConversion
//===----------------------------------------------------------------------===//
class CallOpConversion : public OpConversionPattern<func::CallOp> {
public:
  using Op = func::CallOp;
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ConstantOpConversion
//===----------------------------------------------------------------------===//
class ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ForOpConversion
//===----------------------------------------------------------------------===//
class ForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FuncOpConversion
//===----------------------------------------------------------------------===//
class FuncOpConversion
    : public OpInterfaceConversionPattern<FunctionOpInterface> {
public:
  using OpInterfaceConversionPattern<
      FunctionOpInterface>::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(FunctionOpInterface funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// IndexCastOpConversion
//===----------------------------------------------------------------------===//
class IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ReturnOpConversion
//===----------------------------------------------------------------------===//
class ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using Op = func::ReturnOp;
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

/// Helper to safely get the converter.
static const TypeConverter &getConverter(const TypeConverter *converter) {
  assert(converter && "invalid converter");
  return *converter;
}

//===----------------------------------------------------------------------===//
// CallOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
CallOpConversion::matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  FunctionType nTy = FuncTypeConverter::convertFuncType(
      getConverter(typeConverter), callOp.getCalleeType());

  // Check for ABI attribute override.
  nTy = getABI(callOp, nTy);

  // Don't convert if legal.
  if (callOp.getCalleeType() == nTy)
    return failure();

  SmallVector<Value> operands;
  llvm::append_range(operands, adaptor.getOperands());

  for (auto &&[operand, paramType] :
       llvm::zip_equal(operands, nTy.getInputs())) {
    if (operand.getType() == paramType)
      continue;
    operand = typeConverter->materializeTargetConversion(
        rewriter, callOp.getLoc(), paramType, operand);
  }

  // Create the new call op.
  auto cOp = func::CallOp::create(
      rewriter, callOp.getLoc(), nTy.getResults(), callOp.getCallee(), operands,
      adaptor.getArgAttrsAttr(), adaptor.getArgAttrsAttr(),
      adaptor.getNoInline());
  cOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
  setABI(cOp, nullptr);
  rewriter.replaceOp(callOp, cOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOpConversion
//===----------------------------------------------------------------------===//

LogicalResult ConstantOpConversion::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // This condition makes sure that after the if only index or vector attributes
  // survive.
  if (getTypeConverter()->isLegal(op.getOperation()))
    return failure();
  Type resultType = getTypeConverter()->convertType(op.getType());
  TypedAttr value = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    value = rewriter.getIntegerAttr(
        resultType,
        intAttr.getValue().sextOrTrunc(resultType.getIntOrFloatBitWidth()));
  } else if (auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(value)) {
    if (auto intArrayAttr = dyn_cast<DenseIntElementsAttr>(denseAttr);
        intArrayAttr && isa<IndexType>(intArrayAttr.getElementType())) {
      SmallVector<APInt> intValues;
      Type indexTy =
          getConverter(typeConverter).convertType(rewriter.getIndexType());
      for (const APInt &v : intArrayAttr.getValues<APInt>()) {
        intValues.push_back(v.sextOrTrunc(indexTy.getIntOrFloatBitWidth()));
      }
      value = DenseIntElementsAttr::get(cast<ShapedType>(resultType),
                                        ArrayRef<APInt>(intValues));
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, value);
      return success();
    }
    auto failureOrValues = denseAttr.tryGetValues<APFloat>();
    if (failed(failureOrValues))
      return failure();
    SmallVector<APFloat> values = llvm::to_vector(*failureOrValues);
    value = DenseElementsAttr::get(cast<ShapedType>(resultType), values);
  }
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, value);
  return success();
}

//===----------------------------------------------------------------------===//
// ForOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
ForOpConversion::matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // If no conversion is needed, return failure
  if (getTypeConverter()->isLegal(op.getOperation()))
    return failure();

  // Create the new for operation with empty body
  auto newForOp = scf::ForOp::create(
      rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
      adaptor.getStep(), adaptor.getInitArgs(),
      [](OpBuilder &, Location, Value, ValueRange) {});

  // Inline the region from the old op into the new op
  rewriter.inlineBlockBefore(op.getBody(0), newForOp.getBody(),
                             newForOp.getBody()->end(),
                             newForOp.getBody()->getArguments());

  rewriter.replaceOp(op, newForOp.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
FuncOpConversion::matchAndRewrite(FunctionOpInterface funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  // Don't convert if legal.
  if (getConverter(typeConverter).isLegal(funcOp.getFunctionType()))
    return failure();

  // Check the funcOp has `FunctionType`.
  auto funcTy = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!funcTy) {
    return rewriter.notifyMatchFailure(
        funcOp, "Only support FunctionOpInterface with FunctionType");
  }
  funcTy = getABI(funcOp, funcTy);

  // Convert the signature.
  auto converter = getTypeConverter();
  assert(converter && "invalid converter");
  TypeConverter::SignatureConversion result(funcOp.getNumArguments());
  auto newTy = cast<FunctionType>(FuncTypeConverter::convertFuncType(
      getConverter(typeConverter), funcTy, result));

  // Create the new func op.
  auto newFn = func::FuncOp::create(rewriter, funcOp.getLoc(), funcOp.getName(),
                                    newTy, nullptr, funcOp.getArgAttrsAttr(),
                                    funcOp.getResAttrsAttr());
  rewriter.inlineRegionBefore(funcOp.getFunctionBody(), newFn.getBody(),
                              newFn.end());
  newFn.setVisibility(funcOp.getVisibility());
  newFn->setDiscardableAttrs(funcOp->getDiscardableAttrDictionary());
  setABI(newFn, nullptr);

  // Early exit if it's a declaration.
  if (newFn.isDeclaration()) {
    rewriter.eraseOp(funcOp);
    return success();
  }

  // Convert the signature.
  rewriter.applySignatureConversion(&newFn.getBody().front(), result,
                                    converter);
  rewriter.eraseOp(funcOp);
  return success();
}

//===----------------------------------------------------------------------===//
// IndexCastOpConversion
//===----------------------------------------------------------------------===//

LogicalResult IndexCastOpConversion::matchAndRewrite(
    arith::IndexCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type sourceTy = adaptor.getIn().getType();
  Type targetTy = getTypeConverter()->convertType(op.getOut().getType());
  if (sourceTy == targetTy) {
    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }

  unsigned sourceWidth = sourceTy.getIntOrFloatBitWidth();
  unsigned targetWidth = targetTy.getIntOrFloatBitWidth();

  if (sourceWidth < targetWidth) {
    auto extOp = arith::ExtSIOp::create(rewriter, op.getLoc(), targetTy,
                                        adaptor.getIn());
    rewriter.replaceOp(op, extOp);
    return success();
  } else if (sourceWidth > targetWidth) {
    auto truncOp = arith::TruncIOp::create(rewriter, op.getLoc(), targetTy,
                                           adaptor.getIn());
    rewriter.replaceOp(op, truncOp);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ReturnOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpConversion::matchAndRewrite(func::ReturnOp retOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // Don't convert if legal.
  if (getTypeConverter()->isLegal(retOp.getOperation()))
    return failure();

  // Check for ABI attribute override.
  FunctionType fnTy = getABI(retOp, nullptr);

  // Early exit if no ABI is present.
  if (!fnTy) {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, adaptor.getOperands());
    return success();
  }

  SmallVector<Value> operands;
  llvm::append_range(operands, adaptor.getOperands());

  for (auto &&[operand, paramType] :
       llvm::zip_equal(operands, fnTy.getResults())) {
    if (operand.getType() == paramType)
      continue;
    operand = typeConverter->materializeTargetConversion(
        rewriter, retOp.getLoc(), paramType, operand);
  }

  setABI(retOp, nullptr);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, operands);
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

FunctionType mlir::aster::getABI(Operation *op, FunctionType defaultType) {
  if (auto attr =
          dyn_cast_if_present<TypeAttr>(op->getDiscardableAttr("abi"))) {
    if (auto fnTy = dyn_cast<FunctionType>(attr.getValue()))
      return fnTy;
  }
  return defaultType;
}

void mlir::aster::setABI(Operation *op, FunctionType type) {
  if (!type) {
    op->removeDiscardableAttr("abi");
    return;
  }
  op->setDiscardableAttr("abi", TypeAttr::get(type));
}

void mlir::aster::populateArithConversionPatterns(TypeConverter &converter,
                                                  ConversionTarget &target,
                                                  RewritePatternSet &patterns) {

  target.addDynamicallyLegalOp<
      arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::AddFOp, arith::AndIOp, arith::CeilDivSIOp,arith::CeilDivUIOp,
      arith::CmpFOp, arith::CmpIOp, arith::ConstantOp, arith::DivFOp,
      arith::DivSIOp, arith::DivUIOp, arith::ExtFOp, arith::ExtSIOp,
      arith::ExtUIOp, arith::FloorDivSIOp, arith::IndexCastOp, arith::MaximumFOp, arith::MaxSIOp,
      arith::MaxUIOp, arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp,
      arith::MulFOp, arith::OrIOp, arith::RemSIOp, arith::RemUIOp,
      arith::SelectOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
      arith::SubFOp, arith::TruncFOp, arith::TruncIOp>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  patterns.add<
      GenericOpConversion<arith::AddIOp>, GenericOpConversion<arith::MulIOp>,
      GenericOpConversion<arith::SubIOp>, GenericOpConversion<arith::AddFOp>,
      GenericOpConversion<arith::AndIOp>, GenericOpConversion<arith::CeilDivSIOp>, GenericOpConversion<arith::CeilDivUIOp>, GenericOpConversion<arith::CmpFOp>,
      GenericOpConversion<arith::CmpIOp>, GenericOpConversion<arith::DivFOp>,
      GenericOpConversion<arith::DivSIOp>, GenericOpConversion<arith::DivUIOp>,
      GenericOpConversion<arith::ExtFOp>, GenericOpConversion<arith::ExtSIOp>,
      GenericOpConversion<arith::ExtUIOp>, GenericOpConversion<arith::FloorDivSIOp>,
      GenericOpConversion<arith::MaximumFOp>,
      GenericOpConversion<arith::MaxSIOp>, GenericOpConversion<arith::MaxUIOp>,
      GenericOpConversion<arith::MinimumFOp>,
      GenericOpConversion<arith::MinSIOp>, GenericOpConversion<arith::MinUIOp>,
      GenericOpConversion<arith::MulFOp>, GenericOpConversion<arith::OrIOp>,
      GenericOpConversion<arith::RemSIOp>, GenericOpConversion<arith::RemUIOp>,
      GenericOpConversion<arith::SelectOp>, GenericOpConversion<arith::ShLIOp>,
      GenericOpConversion<arith::ShRSIOp>, GenericOpConversion<arith::ShRUIOp>,
      GenericOpConversion<arith::SubFOp>, GenericOpConversion<arith::TruncFOp>,
      GenericOpConversion<arith::TruncIOp>, ConstantOpConversion,
      IndexCastOpConversion>(converter, patterns.getContext());
}

void mlir::aster::populateFuncConversionPatterns(TypeConverter &converter,
                                                 ConversionTarget &target,
                                                 RewritePatternSet &patterns) {
  target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) -> std::optional<bool> {
        return converter.isLegal(op.getFunctionType());
      });
  patterns.add<CallOpConversion, FuncOpConversion, ReturnOpConversion>(
      converter, patterns.getContext());
  converter.addConversion([&](FunctionType type) {
    return FuncTypeConverter::convertFuncType(converter, type);
  });
}

void mlir::aster::populateScfConversionPatterns(TypeConverter &converter,
                                                ConversionTarget &target,
                                                RewritePatternSet &patterns) {

  target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  patterns.add<GenericOpConversion<scf::YieldOp>, ForOpConversion>(
      converter, patterns.getContext());
}
