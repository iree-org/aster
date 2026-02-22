//===- CodeGenPass.cpp ----------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Passes.h"

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/LSIR/CodeGen/CodeGen.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::aster {
#define GEN_PASS_DEF_CODEGEN
#include "aster/CodeGen/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ForOpConversion
//===----------------------------------------------------------------------===//

/// Custom scf.for conversion that handles SGPR/VGPR type mismatches between
/// init args and iter_arg block arguments by inserting reg_cast
/// materializations.
class ForOpConversion : public OpCodeGenPattern<scf::ForOp> {
public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use the type-based converter for iter_arg types (always VGPR).
    SmallVector<Type> newResultTypes;
    for (Type t : op.getResultTypes()) {
      Type converted = typeConverter->convertType(t);
      if (!converted)
        return failure();
      newResultTypes.push_back(converted);
    }

    // Cast init args to match target types.
    SmallVector<Value> newInitArgs;
    for (auto [initArg, targetType] :
         llvm::zip(adaptor.getInitArgs(), newResultTypes)) {
      if (initArg.getType() == targetType) {
        newInitArgs.push_back(initArg);
      } else {
        newInitArgs.push_back(
            UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
                                               targetType, initArg)
                .getResult(0));
      }
    }

    // Create new scf.for preserving the original index-typed bounds.
    auto newOp = scf::ForOp::create(
        rewriter, op.getLoc(), op.getLowerBound(), op.getUpperBound(),
        op.getStep(), newInitArgs,
        [](OpBuilder &, Location, Value, ValueRange) {});

    // Inline the old body, replacing old block args with new ones.
    rewriter.inlineBlockBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody()->end(),
                               newOp.getBody()->getArguments());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Convert scf.yield operand types to match the parent scf.for iter_arg types.
class YieldOpConversion : public OpCodeGenPattern<scf::YieldOp> {
public:
  using Base::Base;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CodeGen pass
//===----------------------------------------------------------------------===//
struct CodeGen : public aster::impl::CodeGenBase<CodeGen> {
public:
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// CodeGen pass
//===----------------------------------------------------------------------===//

void CodeGen::getDependentDialects(DialectRegistry &registry) const {
  // TODO: Hide these functions behind an interface ala ConvertToLLVM.
  amdgcn::getDependentCodeGenDialects(registry);
  lsir::getDependentCodeGenDialects(registry);
}

void CodeGen::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  FailureOr<ConvertCodeGenState> state = ConvertCodeGenState::create(op);
  if (failed(state))
    return signalPassFailure();
  CodeGenConverter converter(*state);
  ConversionTarget target(getContext());
  // TODO: Hide these functions behind an interface ala ConvertToLLVM.
  amdgcn::populateCodeGenPatterns(converter, patterns, target);
  lsir::populateCodeGenPatterns(converter, patterns, target);
  populateFuncConversionPatterns(converter, target, patterns);
  patterns.add<ForOpConversion, YieldOpConversion>(converter);
  target.addDynamicallyLegalOp<scf::ForOp>(
      [&](scf::ForOp op) { return converter.isLegal(op->getResults()); });
  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
    if (!isa<scf::ForOp>(op->getParentOp()))
      return true;
    return converter.isLegal(op.getOperands());
  });
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
  SmallVector<UnrealizedConversionCastOp> ops;
  getOperation()->walk(
      [&](UnrealizedConversionCastOp castOp) { ops.push_back(castOp); });
  reconcileUnrealizedCasts(ops);
}
