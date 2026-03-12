//===- DecomposeAffineApply.cpp
//--------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wraps upstream affine::decompose() and reorderOperandsByHoistability() into
// an ASTER pass. Splits large affine.apply ops into chains of smaller ops,
// reordered from least to most hoistable. This exposes more opportunities for
// LICM, CSE, and int-range-based optimizations downstream.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster {
#define GEN_PASS_DEF_DECOMPOSEAFFINEAPPLY
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::affine;

namespace {

struct DecomposeAffineApply
    : public aster::impl::DecomposeAffineApplyBase<DecomposeAffineApply> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](AffineApplyOp op) {
      rewriter.setInsertionPoint(op);
      reorderOperandsByHoistability(rewriter, op);
      (void)decompose(rewriter, op,
                      /* mode= */ DecomposeAffineApplyMode::CSEFriendly);
    });
  }
};

} // namespace
