//===- OptimizeAMDGCN.cpp - AMDGCN optimization pass ---------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass applies optimization patterns to AMDGCN dialect operations using
// the greedy pattern rewriter. It includes canonicalization patterns for
// LSIR and AMDGCN dialects and operations.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/IR/RewriteUtils.h"
#include "aster/IR/ValueOrConst.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_OPTIMIZEAMDGCN
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//
struct LoadOpPattern : public OpRewritePattern<amdgcn::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amdgcn::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//
struct StoreOpPattern : public OpRewritePattern<amdgcn::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amdgcn::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// OptimizeAMDGCN pass
//===----------------------------------------------------------------------===//
struct OptimizeAMDGCN
    : public amdgcn::impl::OptimizeAMDGCNBase<OptimizeAMDGCN> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Internal constants and functions
//===----------------------------------------------------------------------===//

/// This is a conservative limit for the constant offset.
constexpr int64_t kMaxConstOffset = 1 << 12;

static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

/// Optimize memory operation offsets by merging ptr_add constant offset into
/// the mem op's constant_offset, and moving ptr_add dynamic offset to the mem
/// op when safe (global memory with SGPR address). Returns success if any
/// changes were made.
/// NOTE: We have to use function references to get the mutable operands because
/// modifying a mutable operand range invalidates the ranges for the other
/// operands.
using GetOperandsFn = llvm::function_ref<MutableOperandRange()>;
static LogicalResult optimizeMemOpOffsets(Operation *op, const InstMetadata *md,
                                          Value addr, Value constantOffset,
                                          Value dynamicOffset,
                                          GetOperandsFn addrMutable,
                                          GetOperandsFn constantOffsetMutable,
                                          GetOperandsFn dynamicOffsetMutable,
                                          PatternRewriter &rewriter) {

  // Get if the address is produced by a ptr_add operation.
  auto ptrAdd = addr.getDefiningOp<amdgcn::PtrAddOp>();
  if (!ptrAdd) {
    return rewriter.notifyMatchFailure(
        op, "expected addr to be produced by ptr_add");
  }

  // Bail if the operation is not a DS or global memory operation.
  if (!md->hasAnyProps({InstProp::Dsmem, InstProp::Global}))
    return rewriter.notifyMatchFailure(
        op, "expected DS or global memory operation");

  int64_t ptrAddOff = ptrAdd.getConstOffset();

  // Get the constant offset from the mem op.
  int32_t memOpOff = 0;
  if (constantOffset) {
    std::optional<int32_t> c = ValueOrI32::getConstant(constantOffset);
    if (!c)
      return rewriter.notifyMatchFailure(op, "expected constant offset");
    memOpOff = *c;
  }

  bool changed = false;
  int64_t newConstOff = ptrAddOff;
  Value newDynOff = ptrAdd.getDynamicOffset();

  // Update the operation on exit if changed. This means setting the addr, and
  // creating a new ptr_add operation.
  // NOTE: This at exit pattern is to de-duplicate the code for setting the addr
  // and the ptr_add operation.
  llvm::scope_exit atExit([&]() {
    if (!changed)
      return;
    ptrAdd = amdgcn::PtrAddOp::create(rewriter, op->getLoc(), ptrAdd.getPtr(),
                                      newDynOff, ptrAdd.getUniformOffset(),
                                      /*const_offset=*/newConstOff);
    rewriter.modifyOpInPlace(op, [&] { addrMutable().assign(ptrAdd); });
  });

  // When possible, merge the constant offset from the ptr_add operation and the
  // mem op.
  int64_t constOff = ptrAddOff + memOpOff;
  if (ptrAddOff != 0 && constOff >= 0 && constOff <= kMaxConstOffset) {
    constantOffsetMutable().assign(
        getI32Constant(rewriter, op->getLoc(), static_cast<int32_t>(constOff)));
    newConstOff = 0;
    changed = true;
  }

  // For DSMem we can only update the constant offset.
  if (md->hasProp(InstProp::Dsmem))
    return success(changed);

  // For global memory we can only update the dynamic offset if the address is
  // not a VGPR.
  bool isVGPRAddr = isVGPR(ptrAdd.getPtr().getType(), 0);

  // If the dynamic offset is not set, and the address is not a VGPR, we can
  // move the dynamic offset to the mem op.
  if (!dynamicOffset && ptrAdd.getDynamicOffset() && !isVGPRAddr) {
    dynamicOffsetMutable().assign(ptrAdd.getDynamicOffset());
    newDynOff = nullptr;
    changed = true;
  }
  return success(changed);
}

//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//

LogicalResult LoadOpPattern::matchAndRewrite(amdgcn::LoadOp op,
                                             PatternRewriter &rewriter) const {
  const InstMetadata *md = op.getInstMetadata();
  assert(md && "expected inst metadata");
  auto getAddr = [&]() -> MutableOperandRange { return op.getAddrMutable(); };
  auto getConstantOffset = [&]() -> MutableOperandRange {
    return op.getConstantOffsetMutable();
  };
  auto getDynamicOffset = [&]() -> MutableOperandRange {
    return op.getDynamicOffsetMutable();
  };
  return optimizeMemOpOffsets(op.getOperation(), md, op.getAddr(),
                              op.getConstantOffset(), op.getDynamicOffset(),
                              getAddr, getConstantOffset, getDynamicOffset,
                              rewriter);
}

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//

LogicalResult StoreOpPattern::matchAndRewrite(amdgcn::StoreOp op,
                                              PatternRewriter &rewriter) const {
  const InstMetadata *md = op.getInstMetadata();
  assert(md && "expected inst metadata");
  auto getAddr = [&]() -> MutableOperandRange { return op.getAddrMutable(); };
  auto getConstantOffset = [&]() -> MutableOperandRange {
    return op.getConstantOffsetMutable();
  };
  auto getDynamicOffset = [&]() -> MutableOperandRange {
    return op.getDynamicOffsetMutable();
  };
  return optimizeMemOpOffsets(op.getOperation(), md, op.getAddr(),
                              op.getConstantOffset(), op.getDynamicOffset(),
                              getAddr, getConstantOffset, getDynamicOffset,
                              rewriter);
}

//===----------------------------------------------------------------------===//
// OptimizeAMDGCN pass
//===----------------------------------------------------------------------===//
void OptimizeAMDGCN::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  // Add canonicalization patterns for LSIR and AMDGCN dialects and operations.
  addCanonicalizationPatterns<lsir::LSIRDialect, amdgcn::AMDGCNDialect>(
      context, patterns);

  // Add custom optimization patterns.
  patterns.add<LoadOpPattern, StoreOpPattern>(context);

  if (failed(applyPatternsAndFoldGreedily(
          op, FrozenRewritePatternSet(std::move(patterns)),
          GreedyRewriteConfig().setUseTopDownTraversal(true)))) {
    return signalPassFailure();
  }
}
