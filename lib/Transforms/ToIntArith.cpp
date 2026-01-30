//===- ToIntArith.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

namespace mlir::aster {
#define GEN_PASS_DEF_TOINTARITH
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ToIntArith pass
//===----------------------------------------------------------------------===//
struct ToIntArith : public aster::impl::ToIntArithBase<ToIntArith> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// ToIntConverter
//===----------------------------------------------------------------------===//
struct ToIntConverter : TypeConverter, aster::FuncTypeConverter {
  ToIntConverter(MLIRContext *ctx, unsigned bitwidth);

private:
  /// Integer type corresponding to index type.
  IntegerType indexType;
};

//===----------------------------------------------------------------------===//
// IdDimOpConversion
//===----------------------------------------------------------------------===//
template <typename Op, typename COp>
class IdDimOpConversion : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ToIntConverter
//===----------------------------------------------------------------------===//

ToIntConverter::ToIntConverter(MLIRContext *ctx, unsigned bitwidth)
    : aster::FuncTypeConverter() {
  indexType = IntegerType::get(ctx, bitwidth);
  Type iTy = indexType;
  addConversion([&](Type type) { return type; });
  addConversion([iTy](IndexType type) { return iTy; });
  // Add generic source and target materializations.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// IdDimOpConversion
//===----------------------------------------------------------------------===//

template <typename Op, typename COp>
LogicalResult IdDimOpConversion<Op, COp>::matchAndRewrite(
    Op op, typename Op::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value res = COp::create(
      rewriter, op.getLoc(),
      static_cast<aster_utils::Dim>(static_cast<int>(op.getDimension())));
  res = arith::IndexCastOp::create(rewriter, op.getLoc(),
                                   rewriter.getIndexType(), res);
  rewriter.replaceOp(op, res);
  return success();
}

//===----------------------------------------------------------------------===//
// ToIntArith pass
//===----------------------------------------------------------------------===//

void ToIntArith::runOnOperation() {
  Operation *op = getOperation();
  auto &dominance = getAnalysis<DominanceInfo>();

  // Helper lambdas for common transformations.
  auto cse = [&]() {
    IRRewriter rewriter(op->getContext());
    eliminateCommonSubExpressions(rewriter, dominance, op, nullptr);
  };
  auto canonicalize = [&](bool canonicalizeAffine) {
    auto arithD = getContext().getLoadedDialect<arith::ArithDialect>();
    auto affineD = getContext().getLoadedDialect<affine::AffineDialect>();

    RewritePatternSet patterns(&getContext());
    arithD->getCanonicalizationPatterns(patterns);
    if (affineD && canonicalizeAffine)
      affineD->getCanonicalizationPatterns(patterns);

    auto config = GreedyRewriteConfig().enableFolding().enableConstantCSE();
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  };

  // Run canonicalization and CSE.
  if (failed(canonicalize(true)))
    return signalPassFailure();
  cse();

  // Lower affine ops.
  {
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    affine::populateAffineExpandIndexOpsPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  // Run canonicalization and CSE.
  if (failed(canonicalize(false)))
    return signalPassFailure();
  cse();

  // Convert index ops to integer ops.
  ToIntConverter converter(&getContext(), bitwidth);
  ConversionTarget target(getContext());
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<arith::ArithDialect, aster_utils::AsterUtilsDialect,
                         ptr::PtrDialect>();
  target.addDynamicallyLegalOp<aster_utils::AssumeRangeOp>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  RewritePatternSet conversionPatterns(&getContext());
  populateFuncConversionPatterns(converter, target, conversionPatterns);
  populateScfConversionPatterns(converter, target, conversionPatterns);
  populateArithConversionPatterns(converter, target, conversionPatterns);
  populatePtrConversionPatterns(converter, target, conversionPatterns);
  conversionPatterns
      .add<IdDimOpConversion<gpu::BlockIdOp, aster_utils::BlockIdOp>,
           IdDimOpConversion<gpu::BlockDimOp, aster_utils::BlockDimOp>,
           IdDimOpConversion<gpu::ThreadIdOp, aster_utils::ThreadIdOp>,
           IdDimOpConversion<gpu::GridDimOp, aster_utils::GridDimOp>,
           GenericOpConversion<aster_utils::AssumeRangeOp>>(converter,
                                                            &getContext());
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          getOperation(), target,
          FrozenRewritePatternSet(std::move(conversionPatterns)), config)))
    return signalPassFailure();

  // Run canonicalization and CSE.
  if (failed(canonicalize(false)))
    return signalPassFailure();
  cse();
}
