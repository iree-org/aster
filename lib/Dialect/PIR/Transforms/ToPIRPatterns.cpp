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

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Dialect/PIR/Transforms/ToPIR.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

namespace {
//===----------------------------------------------------------------------===//
// ArithIPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithIPattern : public OpToPIRPattern<OpTy> {
  using OpToPIRPattern<OpTy>::OpToPIRPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ArithIPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult ArithIPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  rewriter.replaceOpWithNewOp<NewOpTy>(op, dst, adaptor.getLhs(),
                                       adaptor.getRhs());
  return success();
}

//===----------------------------------------------------------------------===//
// ToPIRPass pass
//===----------------------------------------------------------------------===//

static Type convertTypeImpl(Value value, const ToPIRConverter &converter) {
  if (Operation *defOp = value.getDefiningOp();
      defOp && m_Constant().match(value.getDefiningOp()))
    return converter.convertType(value.getType());
  if (isa<RegisterTypeInterface>(value.getType()))
    return value.getType();
  int64_t typeSize = converter.getTypeSize(value.getType());
  int64_t numWords = (typeSize + 3) / 4;
  if (converter.isThreadUniform(value)) {
    return pir::TypedRegisterType::get(
        converter.convertType(value.getType()),
        amdgcn::SGPRRangeType::get(value.getContext(),
                                   RegisterRange(Register(), numWords)));
  }
  return pir::TypedRegisterType::get(
      converter.convertType(value.getType()),
      amdgcn::VGPRRangeType::get(value.getContext(),
                                 RegisterRange(Register(), numWords)));
}

void mlir::aster::pir::populateToPIRPatterns(ToPIRConverter &converter,
                                             RewritePatternSet &patterns,
                                             ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect, pir::PIRDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });

  // Add the type conversions.
  converter.addConversion(
      [&converter](Value value) { return convertTypeImpl(value, converter); });
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Add the patterns.
  patterns.add<ArithIPattern<arith::AddIOp, pir::AddIOp>,
               ArithIPattern<arith::SubIOp, pir::SubIOp>,
               ArithIPattern<arith::MulIOp, pir::MulIOp>,
               ArithIPattern<arith::DivSIOp, pir::DivSIOp>,
               ArithIPattern<arith::DivUIOp, pir::DivUIOp>,
               ArithIPattern<arith::RemSIOp, pir::RemSIOp>,
               ArithIPattern<arith::RemUIOp, pir::RemUIOp>,
               ArithIPattern<arith::AndIOp, pir::AndIOp>,
               ArithIPattern<arith::OrIOp, pir::OrIOp>,
               ArithIPattern<arith::ShLIOp, pir::ShLIOp>,
               ArithIPattern<arith::ShRSIOp, pir::ShRSIOp>,
               ArithIPattern<arith::ShRUIOp, pir::ShRUIOp>>(converter);
}
