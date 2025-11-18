//===- OptimizeArith.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

namespace mlir::aster {
#define GEN_PASS_DEF_OPTIMIZEARITH
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// OptimizeArith pass
//===----------------------------------------------------------------------===//
struct OptimizeArith : public aster::impl::OptimizeArithBase<OptimizeArith> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  LogicalResult
  runGreedyRewriter(std::function<void(RewritePatternSet &)> populatePatterns,
                    bool topDownTraversal = true);
};

//===----------------------------------------------------------------------===//
// Optimization patterns
//===----------------------------------------------------------------------===//

/// Match if the given value is a constant power of two.
static FailureOr<APInt> matchPow2(Value value) {
  APInt valueInt;
  if (!matchPattern(value, m_ConstantInt(&valueInt)))
    return failure();

  if (!valueInt.isPowerOf2())
    return failure();
  return valueInt;
}

// Pattern to convert division by power of two to shift right
template <typename OpTy, typename ShiftOpTy>
struct DivToShift : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    FailureOr<APInt> divisor = matchPow2(op.getRhs());
    if (failed(divisor))
      return failure();
    unsigned shiftAmount = divisor->logBase2();
    auto shiftConst = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                                   op.getType(), shiftAmount);
    rewriter.replaceOpWithNewOp<ShiftOpTy>(op, op.getLhs(), shiftConst);
    return success();
  }
};

// Pattern to convert multiplication by power of two to shift left
struct MulToShift : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    APInt multiplier;
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (matchPattern(rhs, m_ConstantInt(&multiplier))) {
      // rhs is constant
    } else if (matchPattern(lhs, m_ConstantInt(&multiplier))) {
      // lhs is constant, swap
      std::swap(lhs, rhs);
    } else {
      return failure();
    }

    if (!multiplier.isPowerOf2())
      return failure();

    unsigned shiftAmount = multiplier.logBase2();
    auto shiftConst = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                                   op.getType(), shiftAmount);
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, lhs, shiftConst);
    return success();
  }
};

// Pattern to convert remainder by power of two to and operation
template <typename OpTy>
struct RemToAnd : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    FailureOr<APInt> divisor = matchPow2(op.getRhs());
    if (failed(divisor))
      return failure();

    APInt mask = *divisor - 1;
    auto maskConst =
        arith::ConstantIntOp::create(rewriter, op.getLoc(), op.getType(), mask);
    rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(), maskConst);
    return success();
  }
};

/// Get the constant value of a value if available from the data flow solver.
static std::optional<APInt> getMaybeConstantValue(const DataFlowSolver &solver,
                                                  Value value) {
  auto *inferredRange =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(value);
  if (!inferredRange || inferredRange->getValue().isUninitialized())
    return std::nullopt;
  return inferredRange->getValue().getValue().getConstantValue();
}

/// Optimize arith::CmpIOp by replacing it with a constant if the result can be
/// determined from the data flow solver.
struct OptimizeCmpIOp : public OpRewritePattern<arith::CmpIOp> {
  OptimizeCmpIOp(const DataFlowSolver &s, MLIRContext *context)
      : OpRewritePattern<arith::CmpIOp>(context), s(s) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<APInt> c = getMaybeConstantValue(s, op.getResult());
    if (!c.has_value())
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, c->getSExtValue(), 1);
    return success();
  }

  const DataFlowSolver &s;
};
} // namespace

//===----------------------------------------------------------------------===//
// OptimizeArith pass
//===----------------------------------------------------------------------===//

LogicalResult OptimizeArith::runGreedyRewriter(
    std::function<void(RewritePatternSet &)> populatePatterns,
    bool topDownTraversal) {
  RewritePatternSet patterns(&getContext());
  populatePatterns(patterns);

  auto config = GreedyRewriteConfig()
                    .enableFolding()
                    .enableConstantCSE()
                    .setUseTopDownTraversal(true);
  config.setUseTopDownTraversal(topDownTraversal);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config)))
    return failure();
  return success();
}

void OptimizeArith::runOnOperation() {
  Operation *op = getOperation();
  auto &dominance = getAnalysis<DominanceInfo>();

  // Helper lambdas for common transformations.
  auto cse = [&]() {
    IRRewriter rewriter(op->getContext());
    eliminateCommonSubExpressions(rewriter, dominance, op, nullptr);
  };
  auto canonicalize = [&]() {
    auto aD = getContext().getLoadedDialect<arith::ArithDialect>();
    return runGreedyRewriter([&](RewritePatternSet &patterns) {
      aD->getCanonicalizationPatterns(patterns);
    });
  };

  // Helper lambda to get a data flow solver with integer range analysis.
  auto configureSolver = [&](DataFlowSolver &solver) -> LogicalResult {
    dataflow::loadBaselineAnalyses(solver);
    solver.load<dataflow::IntegerRangeAnalysis>();
    return solver.initializeAndRun(op);
  };

  // Helpter lambda to run arith optimizations.
  auto runArithOpt = [&]() {
    return runGreedyRewriter([&](RewritePatternSet &patterns) {
      patterns.add<DivToShift<arith::DivUIOp, arith::ShRUIOp>,
                   DivToShift<arith::DivSIOp, arith::ShRSIOp>, MulToShift,
                   RemToAnd<arith::RemSIOp>, RemToAnd<arith::RemUIOp>>(
          &getContext());
    });
  };

  // Run canonicalization and CSE.
  if (failed(canonicalize()))
    return signalPassFailure();
  cse();

  // Expand ceil/floor div ops.
  {
    RewritePatternSet patterns(&getContext());
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<arith::CeilDivSIOp, arith::CeilDivUIOp,
                        arith::FloorDivSIOp>();
    target.addLegalDialect<arith::ArithDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  // Optimize integer ranges.
  {
    DataFlowSolver solver;
    if (failed(configureSolver(solver)))
      return signalPassFailure();
    if (failed(runGreedyRewriter([&](RewritePatternSet &patterns) {
          arith::populateIntRangeOptimizationsPatterns(patterns, solver);
          patterns.add<OptimizeCmpIOp>(solver, &getContext());
        })))
      return signalPassFailure();
  }

  // Run canonicalization and CSE.
  if (failed(canonicalize()))
    return signalPassFailure();
  cse();

  // Optimize division by constant powers of two.
  if (failed(runArithOpt()))
    return signalPassFailure();

  // Narrow integer ranges to i32 where possible.
  {
    DataFlowSolver solver;
    if (failed(configureSolver(solver)))
      return signalPassFailure();
    if (failed(runGreedyRewriter([&](RewritePatternSet &patterns) {
          arith::populateIntRangeNarrowingPatterns(patterns, solver,
                                                   ArrayRef<unsigned>({32}));
        })))
      return signalPassFailure();
  }

  // Run canonicalization and CSE.
  if (failed(canonicalize()))
    return signalPassFailure();
  cse();

  // Optimize division by constant powers of two.
  if (failed(runArithOpt()))
    return signalPassFailure();
  // Run canonicalization and CSE.
  if (failed(canonicalize()))
    return signalPassFailure();
  cse();

  // Optimize division by constant powers of two.
  if (failed(runArithOpt()))
    return signalPassFailure();

  // Run canonicalization and CSE.
  if (failed(canonicalize()))
    return signalPassFailure();
  cse();
}
