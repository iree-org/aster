//===- ConvertLDSBuffers.cpp - Convert LDS Buffer Operations -------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConvertLDSBuffers pass which converts LDS buffer
// operations to their final form after allocation.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include <string_view>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_CONVERTLDSBUFFERS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct ConvertLDSBuffers
    : public amdgcn::impl::ConvertLDSBuffersBase<ConvertLDSBuffers> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Process a single AllocLDSOp: convert its uses and erase if possible.
  void processAllocOp(RewriterBase &rewriter, amdgcn::AllocLDSOp allocOp);
};
} // namespace

void ConvertLDSBuffers::processAllocOp(RewriterBase &rewriter,
                                       amdgcn::AllocLDSOp allocOp) {
  // Skip allocations without a valid offset.
  std::optional<uint32_t> off = allocOp.getOffset();
  if (!off || ShapedType::isDynamic(allocOp.getStaticSize()))
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(allocOp);

  // Create the constant representing the offset.
  static constexpr std::string_view kAllocLDSTag = "__lds_allocation_size__";
  auto cOffsetOp = arith::ConstantIntOp::create(rewriter, allocOp.getLoc(),
                                                static_cast<int64_t>(*off), 32);
  cOffsetOp->setAttr(kAllocLDSTag,
                     rewriter.getIndexAttr(allocOp.getStaticSize()));

  // Process all uses of the buffer.
  for (Operation *user :
       llvm::make_early_inc_range(allocOp.getBuffer().getUsers())) {
    // Remove dealloc operations.
    if (auto deallocOp = dyn_cast<amdgcn::DeallocLDSOp>(user)) {
      rewriter.eraseOp(deallocOp);
      continue;
    }

    // Replace get_lds_offset operations with the constant offset.
    auto ldsOffOp = dyn_cast<amdgcn::GetLDSOffsetOp>(user);
    if (!ldsOffOp)
      continue;

    rewriter.setInsertionPoint(ldsOffOp);
    Value offset = cOffsetOp.getResult();
    if (cOffsetOp.getType() != ldsOffOp.getResult().getType()) {
      offset = rewriter.create<arith::IndexCastOp>(
          ldsOffOp.getLoc(), ldsOffOp.getResult().getType(),
          cOffsetOp.getResult());
    }
    rewriter.replaceOp(ldsOffOp, offset);
  }

  // Remove the allocation if it has no remaining uses.
  if (allocOp.getBuffer().use_empty())
    rewriter.eraseOp(allocOp);
}

void ConvertLDSBuffers::runOnOperation() {
  Operation *op = getOperation();

  IRRewriter rewriter(op->getContext());
  SmallVector<amdgcn::AllocLDSOp> allocOps;
  op->walk([&](amdgcn::AllocLDSOp allocOp) { allocOps.push_back(allocOp); });
  for (amdgcn::AllocLDSOp allocOp : allocOps)
    processAllocOp(rewriter, allocOp);
}
