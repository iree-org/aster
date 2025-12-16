//===- ToLSIRPatterns.cpp --------------*----------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert to LSIR patterns
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Dialect/LSIR/Transforms/ToLSIR.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::lsir;

namespace {
//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithBinaryOpPattern : public OpToLSIRPattern<OpTy> {
  using OpToLSIRPattern<OpTy>::OpToLSIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithCastOpPattern : public OpToLSIRPattern<OpTy> {
  using OpToLSIRPattern<OpTy>::OpToLSIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//
struct ArithSelectOpPattern : public OpToLSIRPattern<arith::SelectOp> {
  using OpToLSIRPattern::OpToLSIRPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct FromToRegOpPattern : public OpToLSIRPattern<OpTy> {
  using OpToLSIRPattern<OpTy>::OpToLSIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// IDDimOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct IDDimOpPattern : public OpToLSIRPattern<OpTy> {
  using OpToLSIRPattern<OpTy>::OpToLSIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegConstraintPattern
//===----------------------------------------------------------------------===//
struct RegConstraintPattern : public OpToLSIRPattern<RegConstraintOp> {
  using OpToLSIRPattern::OpToLSIRPattern;
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
  rewriter.replaceOpWithNewOp<lsir::SelectOp>(op, dst, adaptor.getCondition(),
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
// IDDimOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult IDDimOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Type regTy = std::is_same_v<OpTy, aster_utils::ThreadIdOp>
                   ? Type(amdgcn::VGPRType::get(op.getContext(), Register()))
                   : Type(amdgcn::SGPRType::get(op.getContext(), Register()));
  auto nOp = NewOpTy::create(
      rewriter, op.getLoc(), regTy,
      static_cast<amdgcn::Dim>(static_cast<int8_t>(op.getDim())));
  rewriter.replaceOpWithNewOp<RegCastOp>(op, type, nOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ToLSIRPass pass
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
  default:
    assert(false && "nyi register kind");
  }
  return nullptr;
}

static Type convertTypeImpl(Value value, const ToLSIRConverter &converter) {
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

static Type convertTypeImpl(Type type, const ToLSIRConverter &converter) {
  if (isa<RegisterTypeInterface>(type))
    return type;
  int64_t typeSize = converter.getTypeSize(type);
  int64_t numWords = (typeSize + 3) / 4;
  return amdgcn::GGPRType::get(
      type.getContext(), RegisterRange(Register(), numWords), std::nullopt);
}

void mlir::aster::lsir::populateToLSIRPatterns(ToLSIRConverter &converter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect, lsir::LSIRDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });
  target.addDynamicallyLegalOp<RegConstraintOp>(
      [&](RegConstraintOp op) { return converter.isLegal(op); });
  target.addIllegalOp<aster_utils::ThreadIdOp, aster_utils::BlockIdOp,
                      aster_utils::BlockDimOp, aster_utils::GridDimOp,
                      lsir::FromRegOp, lsir::ToRegOp, lsir::RegConstraintOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Add the type conversions.
  converter.addConversion(
      [&converter](Type type) { return convertTypeImpl(type, converter); });
  converter.addConversion(
      [&converter](Value value) { return convertTypeImpl(value, converter); });

  populateFuncConversionPatterns(converter, target, patterns);
  // Add the patterns.
  patterns.add<ArithBinaryOpPattern<arith::AddIOp, lsir::AddIOp>,
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
               IDDimOpPattern<aster_utils::ThreadIdOp, amdgcn::ThreadIdOp>,
               IDDimOpPattern<aster_utils::BlockIdOp, amdgcn::BlockIdOp>,
               IDDimOpPattern<aster_utils::BlockDimOp, amdgcn::BlockDimOp>,
               IDDimOpPattern<aster_utils::GridDimOp, amdgcn::GridDimOp>,
               FromToRegOpPattern<ToRegOp>, FromToRegOpPattern<FromRegOp>,
               RegConstraintPattern, ArithSelectOpPattern>(converter);
  patterns.add<GenericOpConversion<RegConstraintOp>>(converter,
                                                     patterns.getContext());
}
