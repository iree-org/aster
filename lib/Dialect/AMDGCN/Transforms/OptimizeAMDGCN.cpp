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
// the greedy pattern rewriter. It includes canonicalization patterns for LSIR
// and AMDGCN dialects, and offset-folding patterns that merge ptr_add and
// lsir.addi constant offsets into memory instruction encoding fields.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/RewriteUtils.h"
#include "aster/IR/ValueOrConst.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
// Pattern declarations
//===----------------------------------------------------------------------===//
struct DSReadPattern : public OpInterfaceRewritePattern<DSReadInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(DSReadInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct DSWritePattern
    : public OpInterfaceRewritePattern<DSWriteInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(DSWriteInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct GlobalLoadPattern
    : public OpInterfaceRewritePattern<GlobalLoadInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct GlobalStorePattern
    : public OpInterfaceRewritePattern<GlobalStoreDwordInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(GlobalStoreDwordInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct BufferLoadPattern
    : public OpInterfaceRewritePattern<BufferLoadInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(BufferLoadInstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct BufferStorePattern
    : public OpInterfaceRewritePattern<BufferStoreInstOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(BufferStoreInstOpInterface op,
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

/// Max constant offset for global memory instructions (13b signed: [0, 4095]).
// TODO: Op properties, not hardcoded.
constexpr int64_t kMaxGlobalConstOffset = (1 << 12) - 1;

/// Max constant offset for DS (LDS) instructions (16b unsigned: [0, 65535]).
constexpr int64_t kMaxDSConstOffset = (1 << 16) - 1;

/// Max constant offset for buffer (MUBUF) instructions (12b unsigned: [0,
/// 4095]).
constexpr int64_t kMaxBufferConstOffset = (1 << 12) - 1;

static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

/// Try to read the current constant offset value from a mem op operand.
/// Returns 0 if the operand is null (absent).
static std::optional<int32_t> getConstOffsetValue(Operand constOff) {
  if (!constOff)
    return 0;
  return ValueOrI32::getConstant(constOff.getValue());
}

/// Try to move the dynamic offset from a ptr_add into a global memory op's
/// offset slot. Only applicable when the base address is SGPR (not VGPR).
/// Returns the dynamic offset value to assign to the mem op, or nullptr if
/// no dynamic offset promotion was performed.
static Value tryPromoteDynamicOffset(Operation *op, amdgcn::PtrAddOp ptrAdd,
                                     int64_t &residualConstOff,
                                     PatternRewriter &rewriter) {
  bool isVGPRBase = isVGPR(ptrAdd.getPtr().getType(), 0);

  // Cannot move dynamic offset if the base address is a VGPR (global ops with
  // VGPR addresses cannot have a separate offset).
  if (isVGPRBase)
    return nullptr;

  Value dynOff = ptrAdd.getDynamicOffset();

  // If there's no dynamic offset and the constant is fully merged, nothing
  // to do.
  if (!dynOff && residualConstOff == 0)
    return nullptr;

  Value cst = getI32Constant(rewriter, op->getLoc(),
                             static_cast<int32_t>(residualConstOff));

  // If we don't have a dynamic offset, materialize the residual constant
  // offset as a VGPR value for the dynamic offset slot.
  if (!dynOff) {
    Value dst = createAllocation(rewriter, op->getLoc(),
                                 getVGPR(rewriter.getContext(), 1));
    dynOff = lsir::MovOp::create(rewriter, op->getLoc(), dst, cst).getDstRes();
    residualConstOff = 0;
    return dynOff;
  }

  // If we have a non-trivial residual constant, fold it into the dynamic
  // offset to avoid i64 additions.
  if (residualConstOff > 0) {
    Value dst = createAllocation(rewriter, op->getLoc(),
                                 getVGPR(rewriter.getContext(), 1));
    dynOff = lsir::AddIOp::create(rewriter, op->getLoc(),
                                  TypeAttr::get(rewriter.getI32Type()), dst,
                                  dynOff, cst)
                 .getDstRes();
    residualConstOff = 0;
  }

  return dynOff;
}

/// Optimize DS and global memory operation offsets by merging ptr_add constant
/// offsets into the mem op's constant_offset field. For global ops with SGPR
/// addresses, also moves the dynamic offset from the ptr_add to the mem op.
///
/// Returns success if any changes were made.
static LogicalResult optimizePtrAddOffsets(Operation *op, Operand addr,
                                           Operand constOffset,
                                           MutableOperand dynOffset,
                                           int64_t maxConstOff,
                                           PatternRewriter &rewriter) {
  auto ptrAdd = addr.getValue().getDefiningOp<amdgcn::PtrAddOp>();
  if (!ptrAdd)
    return rewriter.notifyMatchFailure(
        op, "expected addr to be produced by ptr_add");

  int64_t ptrAddOff = ptrAdd.getConstOffset();

  std::optional<int32_t> memOpOffOpt = getConstOffsetValue(constOffset);
  if (!memOpOffOpt)
    return rewriter.notifyMatchFailure(op, "expected constant offset");
  int32_t memOpOff = *memOpOffOpt;

  bool changed = false;
  int64_t residualConstOff = ptrAddOff;
  Value residualDynOff = ptrAdd.getDynamicOffset();

  // Phase 1: Try to merge the ptr_add constant offset into the mem op.
  int64_t constOff = ptrAddOff + memOpOff;
  if (ptrAddOff != 0 && constOff >= 0 && constOff <= maxConstOff) {
    rewriter.modifyOpInPlace(op, [&] {
      constOffset->set(getI32Constant(rewriter, op->getLoc(),
                                      static_cast<int32_t>(constOff)));
    });
    residualConstOff = 0;
    changed = true;
  }

  // Phase 2: For global ops with SGPR address, move the dynamic offset from the
  // ptr_add to the mem op's offset operand. DS ops have no dynamic offset slot.
  Value newDynOff;
  if (dynOffset) {
    newDynOff = tryPromoteDynamicOffset(op, ptrAdd, residualConstOff, rewriter);
    if (newDynOff) {
      // Clear the ptr_add's dynamic offset since we moved it to the mem op.
      residualDynOff = nullptr;
      changed = true;
    }
  }

  if (!changed)
    return failure();

  // Rebuild the ptr_add with the residual offsets.
  ptr::PtrAddFlags flags = ptrAdd.getFlags();
  auto newPtrAdd = amdgcn::PtrAddOp::create(
      rewriter, op->getLoc(), ptrAdd.getPtr(), residualDynOff,
      ptrAdd.getUniformOffset(), /*const_offset=*/residualConstOff);
  newPtrAdd.setFlags(flags);

  // Update the mem op atomically: set the address and dynamic offset together
  // to avoid transient invalid states (e.g. VGPR address with offset).
  rewriter.modifyOpInPlace(op, [&] {
    addr->set(newPtrAdd);
    if (newDynOff)
      dynOffset.assign(ValueRange{newDynOff});
  });
  return success();
}

namespace {
/// An lsir.addi carrying a constant operand: the non-constant operand and
/// the constant value.
struct AddiMatch {
  Value nonConstantOperand;
  int64_t constant;
};

/// Backward use-def slice of an lsir.addi sub-tree feeding a mem op address.
/// TODO: In the future we may want to collect other ops than just addi but
/// there is a tradeoff between constants coming from mul, shl/r etc having been
/// properly combined at a higher-level (like affine). Refrain from going
/// overboard with low-level analysis unless proven necessary.
struct AddiSlice {
  SetVector<Operation *> addis;
  SmallVector<lsir::AddIOp, 2> constHolders;
};
} // namespace

/// If `a` has a constant i32/index operand, return the non-constant operand
/// and the constant value.
static std::optional<AddiMatch> matchAddiConstOperand(lsir::AddIOp a) {
  APInt c;
  if (matchPattern(a.getRhs(), m_ConstantInt(&c)))
    return AddiMatch{a.getLhs(), c.getSExtValue()};
  if (matchPattern(a.getLhs(), m_ConstantInt(&c)))
    return AddiMatch{a.getRhs(), c.getSExtValue()};
  return std::nullopt;
}

/// Compute the backward use-def slice of `addr` restricted to lsir.addi
/// producers, and record which addis carry a constant operand.
static AddiSlice collectAddiSlice(Value addr) {
  AddiSlice result;
  auto rootAddi = addr.getDefiningOp<lsir::AddIOp>();
  if (!rootAddi)
    return result;

  BackwardSliceOptions opts;
  // Walk only through single-use addis as the rewrite is in-place and changing
  // a multi-use addi's inplace would break its other consumers.
  // TODO: revisit if a non-in-place rewrite becomes desirable.
  Operation *rootOp = rootAddi.getOperation();
  opts.filter = [rootOp](Operation *o) {
    // Allow root unconditionally, it is never modified inplace.
    if (o == rootOp)
      return true;
    auto addi = dyn_cast<lsir::AddIOp>(o);
    return addi && addi.getDstRes().hasOneUse();
  };
  opts.omitBlockArguments = true;
  opts.inclusive = true;
  if (failed(getBackwardSlice(rootOp, &result.addis, opts)))
    return {};

  for (Operation *o : result.addis)
    if (matchAddiConstOperand(cast<lsir::AddIOp>(o)))
      result.constHolders.push_back(cast<lsir::AddIOp>(o));
  return result;
}

/// Optimize memory operations whose address/voffset is produced by an lsir.addi
/// sub-tree containing a constant operand.
///
/// Constant at the root addi (the immediate producer of the address):
///   %addr = lsir.addi i32 %dst, %base, %const
///   amdgcn.mem_op ins(%addr, %data) args(%c0)
/// ->
///   amdgcn.mem_op ins(%base, %data) args(%const)
///
/// Constant at a deeper addi:
///   %p = lsir.addi i32 %dst0, %base, %const          // single use
///   %addr = lsir.addi i32 %dst1, %sgpr_or_vgpr, %p
///   amdgcn.mem_op ins(%addr, %data) args(%c0)
/// ->
///   %addr = lsir.addi i32 %dst1, %sgpr_or_vgpr, %base
///   amdgcn.mem_op ins(%addr, %data) args(%const)
///
/// TODO: multi-constant case -- combine multiple constants from the slice
/// into one merged c() offset (parenthesis-like balance algorithm to keep
/// partial sums within the encoding range).
///
/// Buffer pattern: same shape on the voffset operand.
static LogicalResult optimizeAddiOffsets(Operation *op, Operand foldableOperand,
                                         Operand constOffset,
                                         int64_t maxConstOff,
                                         PatternRewriter &rewriter) {
  Value foldableValue = foldableOperand.getValue();
  if (!foldableValue)
    return rewriter.notifyMatchFailure(op, "no foldable operand value");

  AddiSlice slice = collectAddiSlice(foldableValue);
  if (slice.constHolders.empty())
    return rewriter.notifyMatchFailure(
        op, "no constant-carrying lsir.addi reachable from address");

  std::optional<int32_t> memOpOffOpt = getConstOffsetValue(constOffset);
  if (!memOpOffOpt)
    return rewriter.notifyMatchFailure(op, "expected constant offset");

  // Slice is postorder, so constHolders.front() is the deepest -- the first
  // constant we hit walking back from the address.
  lsir::AddIOp foldTarget = slice.constHolders.front();
  AddiMatch match = *matchAddiConstOperand(foldTarget);
  int64_t mergedOff = match.constant + *memOpOffOpt;
  if (mergedOff < 0 || mergedOff > maxConstOff)
    return rewriter.notifyMatchFailure(op, "merged offset out of range");

  // foldTarget at the root: retarget the mem op's address operand past it.
  // The root may have other users; they still see its unchanged value.
  if (foldTarget.getOperation() == foldableValue.getDefiningOp()) {
    rewriter.modifyOpInPlace(op, [&] {
      foldableOperand->set(match.nonConstantOperand);
      constOffset->set(getI32Constant(rewriter, op->getLoc(),
                                      static_cast<int32_t>(mergedOff)));
    });
    return success();
  }

  // The root is in the backwardSlice via `inclusive=true` regardless of its use
  // count. Check that if the root has other users we bail.
  // TODO: Fix upstream so that backwardSlice inclusive=true does not ignore the
  // filter (or at least have an option to also apply the filter to the root).
  auto root = cast<lsir::AddIOp>(foldableValue.getDefiningOp());
  if (!root.getDstRes().hasOneUse())
    return rewriter.notifyMatchFailure(
        op, "multi-use root would observe wrong value after deep fold");
  OpOperand &use = *foldTarget.getDstRes().use_begin();
  rewriter.modifyOpInPlace(use.getOwner(),
                           [&] { use.set(match.nonConstantOperand); });
  rewriter.modifyOpInPlace(op, [&] {
    constOffset->set(getI32Constant(rewriter, op->getLoc(),
                                    static_cast<int32_t>(mergedOff)));
  });
  return success();
}

//===----------------------------------------------------------------------===//
// DSReadPattern
//===----------------------------------------------------------------------===//

LogicalResult DSReadPattern::matchAndRewrite(DSReadInstOpInterface op,
                                             PatternRewriter &rewriter) const {
  Operation *rawOp = op.getOperation();
  Operand addr = op.getAddress();
  Operand constOff = op.getConstOffsetOperand();

  // Try ptr_add constant merge first.
  if (succeeded(optimizePtrAddOffsets(rawOp, addr, constOff,
                                      /*dynOffset=*/MutableOperand(),
                                      kMaxDSConstOffset, rewriter)))
    return success();

  // Try lsir.addi fold.
  return optimizeAddiOffsets(rawOp, addr, constOff, kMaxDSConstOffset,
                             rewriter);
}

//===----------------------------------------------------------------------===//
// DSWritePattern
//===----------------------------------------------------------------------===//

LogicalResult DSWritePattern::matchAndRewrite(DSWriteInstOpInterface op,
                                              PatternRewriter &rewriter) const {
  Operation *rawOp = op.getOperation();
  Operand addr = op.getAddress();
  Operand constOff = op.getConstOffsetOperand();

  if (succeeded(optimizePtrAddOffsets(rawOp, addr, constOff,
                                      /*dynOffset=*/MutableOperand(),
                                      kMaxDSConstOffset, rewriter)))
    return success();

  return optimizeAddiOffsets(rawOp, addr, constOff, kMaxDSConstOffset,
                             rewriter);
}

//===----------------------------------------------------------------------===//
// GlobalLoadPattern
//===----------------------------------------------------------------------===//

LogicalResult
GlobalLoadPattern::matchAndRewrite(GlobalLoadInstOpInterface op,
                                   PatternRewriter &rewriter) const {
  Operation *rawOp = op.getOperation();
  Operand addr = op.getAddress();
  Operand constOff = op.getConstOffsetOperand();
  MutableOperand dynOffset = op.getOffsetOperand();

  return optimizePtrAddOffsets(rawOp, addr, constOff, dynOffset,
                               kMaxGlobalConstOffset, rewriter);
}

//===----------------------------------------------------------------------===//
// GlobalStorePattern
//===----------------------------------------------------------------------===//

LogicalResult
GlobalStorePattern::matchAndRewrite(GlobalStoreDwordInstOpInterface op,
                                    PatternRewriter &rewriter) const {
  Operation *rawOp = op.getOperation();
  Operand addr = op.getAddress();
  Operand constOff = op.getConstOffsetOperand();
  MutableOperand dynOffset = op.getOffsetOperand();

  return optimizePtrAddOffsets(rawOp, addr, constOff, dynOffset,
                               kMaxGlobalConstOffset, rewriter);
}

//===----------------------------------------------------------------------===//
// BufferLoadPattern
//===----------------------------------------------------------------------===//

LogicalResult
BufferLoadPattern::matchAndRewrite(BufferLoadInstOpInterface op,
                                   PatternRewriter &rewriter) const {
  // TODO: Improve the pattern and remove this condition.
  if (op.getLds())
    return failure();
  Operation *rawOp = op.getOperation();
  MutableOperand offOrIdx = op.getOffOrIdxOperand();
  if (!offOrIdx || offOrIdx.empty())
    return rewriter.notifyMatchFailure(rawOp, "no off_or_idx operand");

  Operand voffset = offOrIdx[0];
  Operand constOff = op.getConstOffsetOperand();

  return optimizeAddiOffsets(rawOp, voffset, constOff, kMaxBufferConstOffset,
                             rewriter);
}

//===----------------------------------------------------------------------===//
// BufferStorePattern
//===----------------------------------------------------------------------===//

LogicalResult
BufferStorePattern::matchAndRewrite(BufferStoreInstOpInterface op,
                                    PatternRewriter &rewriter) const {
  Operation *rawOp = op.getOperation();
  MutableOperand offOrIdx = op.getOffOrIdxOperand();
  if (!offOrIdx || offOrIdx.empty())
    return rewriter.notifyMatchFailure(rawOp, "no off_or_idx operand");

  Operand voffset = offOrIdx[0];
  Operand constOff = op.getConstOffsetOperand();

  return optimizeAddiOffsets(rawOp, voffset, constOff, kMaxBufferConstOffset,
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

  // Add offset-folding patterns for each memory instruction family.
  patterns.add<DSReadPattern, DSWritePattern, GlobalLoadPattern,
               GlobalStorePattern, BufferLoadPattern, BufferStorePattern>(
      context);

  if (failed(applyPatternsGreedily(
          op, FrozenRewritePatternSet(std::move(patterns)),
          GreedyRewriteConfig().setUseTopDownTraversal(true))))
    return signalPassFailure();
}
