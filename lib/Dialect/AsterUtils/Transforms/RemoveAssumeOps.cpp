//===- RemoveAssumeOps.cpp - Remove assume and passthrough ops ------------===//
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
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Regex.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_REMOVEASSUMEOPS
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// RemoveAssumeOps pass
//===----------------------------------------------------------------------===//
struct RemoveAssumeOps
    : public aster_utils::impl::RemoveAssumeOpsBase<RemoveAssumeOps> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void RemoveAssumeOps::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(&getContext());

  // Parse tag-regex option.
  llvm::Regex tagRegexObj;
  bool useTagRegex = false;
  if (!tagRegex.empty()) {
    tagRegexObj = llvm::Regex(tagRegex);
    if (!tagRegexObj.isValid()) {
      emitError(module.getLoc()) << "invalid tag-regex: " << tagRegex;
      return signalPassFailure();
    }
    useTagRegex = true;
  }

  module.walk([&](Operation *op) {
    // Remove assume_uniform ops.
    if (removeUniform && isa<AssumeUniformOp>(op)) {
      rewriter.replaceOp(op, cast<AssumeUniformOp>(op).getInput());
      return;
    }

    // Remove assume_range ops.
    if (removeRange && isa<AssumeRangeOp>(op)) {
      rewriter.replaceOp(op, cast<AssumeRangeOp>(op).getInput());
      return;
    }

    // Remove passthrough ops.
    if (auto passOp = dyn_cast<PassthroughOp>(op)) {
      bool shouldRemove = false;
      if (removePassthrough) {
        shouldRemove = true;
      } else if (useTagRegex) {
        if (std::optional<llvm::StringRef> tag = passOp.getTag())
          shouldRemove = tagRegexObj.match(*tag);
      }
      if (shouldRemove)
        rewriter.replaceOp(op, passOp.getInput());
    }
  });
}
