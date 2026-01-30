//===- Legalizer.cpp - Legalize operations for code generation ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "aster/CodeGen/Passes.h"

#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
#define GEN_PASS_DEF_LEGALIZER
#include "aster/CodeGen/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// Legalizer pass
//===----------------------------------------------------------------------===//
struct Legalizer : public aster::impl::LegalizerBase<Legalizer> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Legalizer pass
//===----------------------------------------------------------------------===//

static LogicalResult
runGreedyRewriter(Operation *op,
                  std::function<void(RewritePatternSet &)> populatePatterns,
                  bool topDownTraversal = true) {
  RewritePatternSet patterns(op->getContext());
  populatePatterns(patterns);
  auto config = GreedyRewriteConfig()
                    .enableFolding()
                    .enableConstantCSE()
                    .setUseTopDownTraversal(topDownTraversal);
  if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
    return failure();
  return success();
}

void Legalizer::runOnOperation() {
  Operation *op = getOperation();

  // Legalize operations that don't require type conversion.
  if (failed(runGreedyRewriter(op, [&](RewritePatternSet &patterns) {
        amdgcn::populateAMDGPULegalizationPatterns(patterns);
      })))
    return signalPassFailure();

  // Legalize operations that require type conversion.
  {
    LegalizerTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    RewritePatternSet conversionPatterns(&getContext());
    ConversionConfig config;
    config.allowPatternRollback = false;
    amdgcn::populateAMDGPUTypeLegalizationPatterns(converter, target,
                                                   conversionPatterns);
    if (failed(applyPartialConversion(
            getOperation(), target,
            FrozenRewritePatternSet(std::move(conversionPatterns)), config)))
      return signalPassFailure();
  }
}
