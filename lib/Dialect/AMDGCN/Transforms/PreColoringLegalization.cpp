//===- PreColoringLegalization.cpp - Legalize ops before register coloring ===//
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
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "amdgcn-pre-coloring-legalization"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_PRECOLORINGLEGALIZATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// PreColoringLegalization pass
//===----------------------------------------------------------------------===//
struct PreColoringLegalization
    : public amdgcn::impl::PreColoringLegalizationBase<
          PreColoringLegalization> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Legalization helpers
//===----------------------------------------------------------------------===//

// Legalize lsir.copy where the target is an SCC register and the source is an
// SGPR into s_cmp_eq_u32, setting SCC to (sgpr == 1).
static void legalizeScalarCopyToSCC(lsir::CopyOp copyOp, IRRewriter &rewriter) {
  Value target = copyOp.getTarget();
  Value source = copyOp.getSource();
  if (!isa<SCCType>(target.getType()) || !isa<SGPRType>(source.getType()))
    return;

  rewriter.setInsertionPoint(copyOp);
  Location loc = copyOp.getLoc();
  Value c1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  SCmpEqU32::create(rewriter, loc, target, source, c1);
  rewriter.eraseOp(copyOp);
}

//===----------------------------------------------------------------------===//
// PreColoringLegalization pass
//===----------------------------------------------------------------------===//
void PreColoringLegalization::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());
  op->walk(
      [&](lsir::CopyOp copyOp) { legalizeScalarCopyToSCC(copyOp, rewriter); });
}
