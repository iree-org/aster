//===- FromAnyToPoison.cpp - Convert FromAnyOp to ub.poison --------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_FROMANYTOPOISON
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// FromAnyToPoison pattern
//===----------------------------------------------------------------------===//

struct FromAnyToPoisonPattern : public OpRewritePattern<FromAnyOp> {
  using OpRewritePattern<FromAnyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromAnyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ub::PoisonOp>(op, op.getType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FromAnyToPoison pass
//===----------------------------------------------------------------------===//

struct FromAnyToPoison
    : public aster_utils::impl::FromAnyToPoisonBase<FromAnyToPoison> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void FromAnyToPoison::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFromAnyToPoisonPattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

void aster_utils::populateFromAnyToPoisonPattern(RewritePatternSet &patterns) {
  patterns.add<FromAnyToPoisonPattern>(patterns.getContext());
}
