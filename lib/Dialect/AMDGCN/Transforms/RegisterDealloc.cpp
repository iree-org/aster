//===- RegisterDealloc.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REGISTERDEALLOC
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// RegisterDealloc pass
//===----------------------------------------------------------------------===//
struct RegisterDealloc
    : public amdgcn::impl::RegisterDeallocBase<RegisterDealloc> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// DeallocTypeConverter
//===----------------------------------------------------------------------===//
struct DeallocTypeConverter : public ::mlir::TypeConverter, FuncTypeConverter {
  DeallocTypeConverter(MLIRContext *ctx);
};

//===----------------------------------------------------------------------===//
//
// Conversion patterns
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

class AllocaOpConversion : public OpConversionPattern<AllocaOp> {
public:
  using OpAdaptor = AllocaOp::Adaptor;
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// InstOpConversion
//===----------------------------------------------------------------------===//

class InstOpConversion : public OpInterfaceConversionPattern<InstOpInterface> {
public:
  using OpInterfaceConversionPattern<
      InstOpInterface>::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(InstOpInterface op, ArrayRef<Value> adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOpConversion
//===----------------------------------------------------------------------===//

class MakeRegisterRangeOpConversion
    : public OpConversionPattern<MakeRegisterRangeOp> {
public:
  using OpAdaptor = MakeRegisterRangeOp::Adaptor;
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MakeRegisterRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOpConversion
//===----------------------------------------------------------------------===//

class SplitRegisterRangeOpConversion
    : public OpConversionPattern<SplitRegisterRangeOp> {
public:
  using OpAdaptor = SplitRegisterRangeOp::Adaptor;
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SplitRegisterRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// DeallocTypeConverter
//===----------------------------------------------------------------------===//

DeallocTypeConverter::DeallocTypeConverter(MLIRContext *ctx) {
  addConversion(+[](Type type) { return type; });
  addConversion(+[](RegisterTypeInterface type) {
    return type.cloneRegisterType(type.getAsRange().getAsValueRange());
  });
}

//===----------------------------------------------------------------------===//
// AllocaOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpConversion::matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  if (getTypeConverter()->isLegal(op.getOperation()))
    return failure();
  Type newType = getTypeConverter()->convertType(op.getType());
  (void)rewriter.replaceOpWithNewOp<AllocaOp>(op, newType);
  return success();
}

//===----------------------------------------------------------------------===//
// InstOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
InstOpConversion::matchAndRewrite(InstOpInterface op, ArrayRef<Value> adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (getTypeConverter()->isLegal(op))
    return failure();
  // TODO: This is a hack, add a propper way to handle operands and results.
  Operation *clonedOp = rewriter.clone(*op.getOperation());
  clonedOp->setOperands(adaptor);
  for (OpResult res : clonedOp->getResults()) {
    Type newType = getTypeConverter()->convertType(res.getType());
    if (!newType)
      return failure();
    res.setType(newType);
  }
  rewriter.replaceOp(op, clonedOp);
  return success();
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangeOpConversion
//===----------------------------------------------------------------------===//

LogicalResult MakeRegisterRangeOpConversion::matchAndRewrite(
    MakeRegisterRangeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (getTypeConverter()->isLegal(op.getType()))
    return failure();
  rewriter.replaceOpWithNewOp<MakeRegisterRangeOp>(op, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// SplitRegisterRangeOpConversion
//===----------------------------------------------------------------------===//

LogicalResult SplitRegisterRangeOpConversion::matchAndRewrite(
    SplitRegisterRangeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (getTypeConverter()->isLegal(op.getOperation()))
    return failure();
  rewriter.replaceOpWithNewOp<SplitRegisterRangeOp>(op, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// RegisterDealloc pass
//===----------------------------------------------------------------------===//

void RegisterDealloc::runOnOperation() {
  DeallocTypeConverter converter(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<amdgcn::AMDGCNDialect>();
  // TODO: Fix upstream to not have to include every single op.
  target.addDynamicallyLegalOp<
      amdgcn::AllocaOp, amdgcn::MakeRegisterRangeOp,
      amdgcn::SplitRegisterRangeOp, amdgcn::inst::VOP1Op, amdgcn::inst::VOP2Op,
      amdgcn::inst::VOP3PMAIOp, amdgcn::inst::VOP3PScaledMAIOp, amdgcn::LoadOp,
      amdgcn::StoreOp>([&](Operation *op) -> std::optional<bool> {
    return converter.isLegal(op);
  });
  RewritePatternSet conversionPatterns(&getContext());
  conversionPatterns
      .add<AllocaOpConversion, InstOpConversion, MakeRegisterRangeOpConversion,
           SplitRegisterRangeOpConversion>(converter, &getContext());
  populateFuncConversionPatterns(converter, target, conversionPatterns);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          getOperation(), target,
          FrozenRewritePatternSet(std::move(conversionPatterns)), config)))
    return signalPassFailure();
}
