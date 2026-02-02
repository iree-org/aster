//===- CodeGenPatterns.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSIR CodeGen patterns
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/LSIR/CodeGen/CodeGen.h"

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::lsir;

namespace {
//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithBinaryOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithCastOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//
struct ArithSelectOpPattern : public OpCodeGenPattern<arith::SelectOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct FromToRegOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegConstraintPattern
//===----------------------------------------------------------------------===//
struct RegConstraintPattern : public OpCodeGenPattern<RegConstraintOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AssumeRangeOpPattern
//===----------------------------------------------------------------------===//
struct AssumeRangeOpPattern
    : public OpCodeGenPattern<aster_utils::AssumeRangeOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult ArithBinaryOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  rewriter.replaceOpWithNewOp<NewOpTy>(op, TypeAttr::get(op.getType()), dst,
                                       adaptor.getLhs(), adaptor.getRhs());
  return success();
}

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult ArithCastOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);

  // Get the element type from the original operation
  Type srcElemType = getElementTypeOrSelf(op.getIn().getType());
  Type dstElemType = getElementTypeOrSelf(op.getType());

  rewriter.replaceOpWithNewOp<NewOpTy>(op, TypeAttr::get(dstElemType),
                                       TypeAttr::get(srcElemType), dst,
                                       adaptor.getIn());
  return success();
}

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ArithSelectOpPattern::matchAndRewrite(
    arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  rewriter.replaceOpWithNewOp<lsir::SelectOp>(op, dst, adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult FromToRegOpPattern<OpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  // If the input is a constant, create a mov to the proper register type.
  if (m_Constant().match(input.getDefiningOp())) {
    Type type = this->converter.convertType(op);
    Value dst = this->createAlloca(rewriter, op.getLoc(), type);
    rewriter.replaceOpWithNewOp<lsir::MovOp>(op, dst, input);
    return success();
  }
  rewriter.replaceOp(op, input);
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::aster::lsir::getDependentCodeGenDialects(DialectRegistry &registry) {
  registry.insert<lsir::LSIRDialect>();
}

void mlir::aster::lsir::populateCodeGenPatterns(CodeGenConverter &converter,
                                                RewritePatternSet &patterns,
                                                ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<lsir::LSIRDialect>();
  target.addIllegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });
  target.addDynamicallyLegalOp<RegConstraintOp>(
      [&](RegConstraintOp op) { return converter.isLegal(op); });
  target.addIllegalOp<aster_utils::AssumeRangeOp, lsir::FromRegOp,
                      lsir::ToRegOp, lsir::RegConstraintOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  // Add the patterns.
  patterns
      .add<ArithBinaryOpPattern<arith::AddIOp, lsir::AddIOp>,
           ArithBinaryOpPattern<arith::SubIOp, lsir::SubIOp>,
           ArithBinaryOpPattern<arith::MulIOp, lsir::MulIOp>,
           ArithBinaryOpPattern<arith::DivSIOp, lsir::DivSIOp>,
           ArithBinaryOpPattern<arith::DivUIOp, lsir::DivUIOp>,
           ArithBinaryOpPattern<arith::RemSIOp, lsir::RemSIOp>,
           ArithBinaryOpPattern<arith::RemUIOp, lsir::RemUIOp>,
           ArithBinaryOpPattern<arith::AndIOp, lsir::AndIOp>,
           ArithBinaryOpPattern<arith::OrIOp, lsir::OrIOp>,
           ArithBinaryOpPattern<arith::XOrIOp, lsir::XOrIOp>,
           ArithBinaryOpPattern<arith::ShLIOp, lsir::ShLIOp>,
           ArithBinaryOpPattern<arith::ShRSIOp, lsir::ShRSIOp>,
           ArithBinaryOpPattern<arith::ShRUIOp, lsir::ShRUIOp>,
           ArithBinaryOpPattern<arith::MaxSIOp, lsir::MaxSIOp>,
           ArithBinaryOpPattern<arith::MaxUIOp, lsir::MaxUIOp>,
           ArithBinaryOpPattern<arith::AddFOp, lsir::AddFOp>,
           ArithBinaryOpPattern<arith::SubFOp, lsir::SubFOp>,
           ArithBinaryOpPattern<arith::MulFOp, lsir::MulFOp>,
           ArithBinaryOpPattern<arith::DivFOp, lsir::DivFOp>,
           ArithBinaryOpPattern<arith::MaximumFOp, lsir::MaximumFOp>,
           ArithBinaryOpPattern<arith::MinimumFOp, lsir::MinimumFOp>,
           ArithCastOpPattern<arith::ExtSIOp, lsir::ExtSIOp>,
           ArithCastOpPattern<arith::ExtUIOp, lsir::ExtUIOp>,
           ArithCastOpPattern<arith::TruncIOp, lsir::TruncIOp>,
           ArithCastOpPattern<arith::ExtFOp, lsir::ExtFOp>,
           ArithCastOpPattern<arith::TruncFOp, lsir::TruncFOp>,
           ArithCastOpPattern<arith::FPToSIOp, lsir::FPToSIOp>,
           ArithCastOpPattern<arith::FPToUIOp, lsir::FPToUIOp>,
           ArithCastOpPattern<arith::SIToFPOp, lsir::SIToFPOp>,
           ArithCastOpPattern<arith::UIToFPOp, lsir::UIToFPOp>,
           FromToRegOpPattern<ToRegOp>, FromToRegOpPattern<FromRegOp>,
           RegConstraintPattern, AssumeRangeOpPattern, ArithSelectOpPattern>(
          converter);
  patterns.add<GenericOpConversion<RegConstraintOp>>(converter,
                                                     patterns.getContext());
}
