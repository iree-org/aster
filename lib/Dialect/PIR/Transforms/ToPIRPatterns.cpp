//===- ToPIRPatterns.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert to PIR patterns
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Dialect/PIR/Transforms/ToPIR.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

namespace {
//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithBinaryOpPattern : public OpToPIRPattern<OpTy> {
  using OpToPIRPattern<OpTy>::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithCastOpPattern : public OpToPIRPattern<OpTy> {
  using OpToPIRPattern<OpTy>::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//
struct ArithSelectOpPattern : public OpToPIRPattern<arith::SelectOp> {
  using OpToPIRPattern::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct FromToRegOpPattern : public OpToPIRPattern<OpTy> {
  using OpToPIRPattern<OpTy>::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct IDDimOpPattern : public OpToPIRPattern<OpTy> {
  using OpToPIRPattern<OpTy>::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegConstraintPattern
//===----------------------------------------------------------------------===//
struct RegConstraintPattern : public OpToPIRPattern<RegConstraintOp> {
  using OpToPIRPattern::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
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
  rewriter.replaceOpWithNewOp<pir::SelectOp>(op, dst, adaptor.getCondition(),
                                             adaptor.getTrueValue(),
                                             adaptor.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult FromToRegOpPattern<OpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOp(op, adaptor.getInput());
  return success();
}

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult IDDimOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Type regTy = std::is_same_v<OpTy, pir::ThreadIdOp>
                   ? Type(amdgcn::VGPRType::get(op.getContext(), Register()))
                   : Type(amdgcn::SGPRType::get(op.getContext(), Register()));
  auto nOp = NewOpTy::create(
      rewriter, op.getLoc(), regTy,
      static_cast<amdgcn::Dim>(static_cast<int8_t>(op.getDim())));
  rewriter.replaceOpWithNewOp<RegCastOp>(op, type, nOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ToPIRPass pass
//===----------------------------------------------------------------------===//

/// Untagle unrealized conversion casts to find the original value.
static Value untagleConvertValue(Value value) {
  if (isa<RegisterTypeInterface>(value.getType()))
    return value;
  auto cOp =
      dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  while (cOp && cOp.getNumOperands() == 1) {
    Value value = cOp.getOperand(0);
    if (isa<RegisterTypeInterface>(value.getType()))
      return value;
    cOp =
        dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  }
  return value;
}

static Type convertAttrConstraintToType(Attribute constraint,
                                        int64_t numWords) {
  auto kind = dyn_cast<amdgcn::RegisterKindAttr>(constraint);
  if (!kind)
    return nullptr;
  switch (kind.getValue()) {
  case amdgcn::RegisterKind::SGPR:
    if (numWords == 1)
      return amdgcn::SGPRType::get(kind.getContext(), Register());
    return amdgcn::SGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::VGPR:
    if (numWords == 1)
      return amdgcn::VGPRType::get(kind.getContext(), Register());
    return amdgcn::VGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  case amdgcn::RegisterKind::AGPR:
    if (numWords == 1)
      return amdgcn::AGPRType::get(kind.getContext(), Register());
    return amdgcn::AGPRRangeType::get(kind.getContext(),
                                      RegisterRange(Register(), numWords));
  }
  return nullptr;
}

static Type convertTypeImpl(Value value, const ToPIRConverter &converter) {
  if (Operation *defOp = value.getDefiningOp();
      defOp && m_Constant().match(value.getDefiningOp()))
    return value.getType();
  value = untagleConvertValue(value);
  if (isa<RegisterTypeInterface>(value.getType()))
    return value.getType();

  int64_t typeSize = converter.getTypeSize(value.getType());
  int64_t numWords = (typeSize + 3) / 4;

  // If there is a register constraint, use it to determine the type.
  if (Attribute constraint =
          converter.getState().getRegisterConstraint(value)) {
    if (Type t = convertAttrConstraintToType(constraint, numWords))
      return t;
  }

  std::optional<bool> isUniform = converter.isThreadUniform(value);
  assert(isUniform.has_value() &&
         "Type conversion for value without known thread-uniformity");
  return amdgcn::GGPRType::get(value.getContext(),
                               RegisterRange(Register(), numWords), isUniform);
}

static Type convertTypeImpl(Type type, const ToPIRConverter &converter) {
  if (isa<RegisterTypeInterface>(type))
    return type;
  int64_t typeSize = converter.getTypeSize(type);
  int64_t numWords = (typeSize + 3) / 4;
  return amdgcn::GGPRType::get(
      type.getContext(), RegisterRange(Register(), numWords), std::nullopt);
}

void mlir::aster::pir::populateToPIRPatterns(ToPIRConverter &converter,
                                             RewritePatternSet &patterns,
                                             ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect, pir::PIRDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });
  target.addDynamicallyLegalOp<RegConstraintOp>(
      [&](RegConstraintOp op) { return converter.isLegal(op); });
  target.addIllegalOp<pir::ThreadIdOp, pir::BlockIdOp, pir::BlockDimOp,
                      pir::GridDimOp,
                      pir::FromRegOp, pir::ToRegOp, pir::RegConstraintOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Add the type conversions.
  converter.addConversion(
      [&converter](Type type) { return convertTypeImpl(type, converter); });
  converter.addConversion(
      [&converter](Value value) { return convertTypeImpl(value, converter); });

  populateFuncConversionPatterns(converter, target, patterns);
  // Add the patterns.
  patterns.add<ArithBinaryOpPattern<arith::AddIOp, pir::AddIOp>,
               ArithBinaryOpPattern<arith::SubIOp, pir::SubIOp>,
               ArithBinaryOpPattern<arith::MulIOp, pir::MulIOp>,
               ArithBinaryOpPattern<arith::DivSIOp, pir::DivSIOp>,
               ArithBinaryOpPattern<arith::DivUIOp, pir::DivUIOp>,
               ArithBinaryOpPattern<arith::RemSIOp, pir::RemSIOp>,
               ArithBinaryOpPattern<arith::RemUIOp, pir::RemUIOp>,
               ArithBinaryOpPattern<arith::AndIOp, pir::AndIOp>,
               ArithBinaryOpPattern<arith::OrIOp, pir::OrIOp>,
               ArithBinaryOpPattern<arith::XOrIOp, pir::XOrIOp>,
               ArithBinaryOpPattern<arith::ShLIOp, pir::ShLIOp>,
               ArithBinaryOpPattern<arith::ShRSIOp, pir::ShRSIOp>,
               ArithBinaryOpPattern<arith::ShRUIOp, pir::ShRUIOp>,
               ArithBinaryOpPattern<arith::MaxSIOp, pir::MaxSIOp>,
               ArithBinaryOpPattern<arith::MaxUIOp, pir::MaxUIOp>,
               ArithBinaryOpPattern<arith::AddFOp, pir::AddFOp>,
               ArithBinaryOpPattern<arith::SubFOp, pir::SubFOp>,
               ArithBinaryOpPattern<arith::MulFOp, pir::MulFOp>,
               ArithBinaryOpPattern<arith::DivFOp, pir::DivFOp>,
               ArithBinaryOpPattern<arith::MaximumFOp, pir::MaximumFOp>,
               ArithBinaryOpPattern<arith::MinimumFOp, pir::MinimumFOp>,
               ArithCastOpPattern<arith::ExtSIOp, pir::ExtSIOp>,
               ArithCastOpPattern<arith::ExtUIOp, pir::ExtUIOp>,
               ArithCastOpPattern<arith::TruncIOp, pir::TruncIOp>,
               ArithCastOpPattern<arith::ExtFOp, pir::ExtFOp>,
               ArithCastOpPattern<arith::TruncFOp, pir::TruncFOp>,
               ArithCastOpPattern<arith::FPToSIOp, pir::FPToSIOp>,
               ArithCastOpPattern<arith::FPToUIOp, pir::FPToUIOp>,
               ArithCastOpPattern<arith::SIToFPOp, pir::SIToFPOp>,
               ArithCastOpPattern<arith::UIToFPOp, pir::UIToFPOp>,
               IDDimOpPattern<pir::ThreadIdOp, amdgcn::ThreadIdOp>,
               IDDimOpPattern<pir::BlockIdOp, amdgcn::BlockIdOp>,
               IDDimOpPattern<pir::BlockDimOp, amdgcn::BlockDimOp>,
               IDDimOpPattern<pir::GridDimOp, amdgcn::GridDimOp>,
               FromToRegOpPattern<ToRegOp>, FromToRegOpPattern<FromRegOp>,
               RegConstraintPattern, ArithSelectOpPattern>(converter);
  patterns.add<GenericOpConversion<RegConstraintOp>>(converter,
                                                     patterns.getContext());
}
