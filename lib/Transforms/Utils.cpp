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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"

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

  // Don't convert if legal.
  if (callOp.getCalleeType() == nTy)
    return failure();

  // Create the new call op.
  auto cOp = func::CallOp::create(
      rewriter, callOp.getLoc(), nTy.getResults(), callOp.getCallee(),
      adaptor.getOperands(), adaptor.getArgAttrsAttr(),
      adaptor.getArgAttrsAttr(), adaptor.getNoInline());
  cOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
  rewriter.replaceOp(callOp, cOp);
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

  // Convert the signature.
  auto converter = getTypeConverter();
  assert(converter && "invalid converter");
  TypeConverter::SignatureConversion result(funcOp.getNumArguments());
  auto newTy = cast<FunctionType>(FuncTypeConverter::convertFunctionSignature(
      getConverter(typeConverter), funcOp, result));

  // Create the new func op.
  auto newFn = func::FuncOp::create(rewriter, funcOp.getLoc(), funcOp.getName(),
                                    newTy, nullptr, funcOp.getArgAttrsAttr(),
                                    funcOp.getResAttrsAttr());
  rewriter.inlineRegionBefore(funcOp.getFunctionBody(), newFn.getBody(),
                              newFn.end());
  newFn.setVisibility(funcOp.getVisibility());

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
// ReturnOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpConversion::matchAndRewrite(func::ReturnOp retOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // Don't convert if legal.
  if (getTypeConverter()->isLegal(retOp.getOperation()))
    return failure();
  rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

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
