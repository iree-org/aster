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
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/RewriteUtils.h"
#include "aster/IR/ValueOrConst.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include <functional>

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

/// Max cst offset for global memory instructions (13b signed: [-4096, 4095]).
// TODO: Op properties, not hardcoded.
constexpr int64_t kMaxGlobalConstOffset = (1 << 12) - 1;

/// Max cst offset for DS (LDS) instructions (16b unsigned: [0, 65535]).
constexpr int64_t kMaxDSConstOffset = (1 << 16) - 1;

/// Max cst offset for buffer (MUBUF) instructions (12b unsigned: [0, 4095]).
constexpr int64_t kMaxBufferConstOffset = (1 << 12) - 1;

/// Return the maximum constant offset for the given instruction type.
static int64_t getMaxConstOffset(const InstMetadata *md) {
  if (md->hasProp(InstProp::Dsmem))
    return kMaxDSConstOffset;
  if (md->hasProp(InstProp::Buffer))
    return kMaxBufferConstOffset;
  return kMaxGlobalConstOffset;
}

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
  // Buffer ops use optimizeAddiOffsets instead (no ptr_add addressing).
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
    ptr::PtrAddFlags flags = ptrAdd.getFlags();
    ptrAdd = amdgcn::PtrAddOp::create(rewriter, op->getLoc(), ptrAdd.getPtr(),
                                      newDynOff, ptrAdd.getUniformOffset(),
                                      /*const_offset=*/newConstOff);
    ptrAdd.setFlags(flags);
    rewriter.modifyOpInPlace(op, [&] { addrMutable().assign(ptrAdd); });
  });

  // Use the correct max offset for the instruction type.
  int64_t maxConstOff = getMaxConstOffset(md);

  // When possible, merge the constant offset from the ptr_add operation and the
  // mem op.
  int64_t constOff = ptrAddOff + memOpOff;
  if (ptrAddOff != 0 && constOff >= 0 && constOff <= maxConstOff) {
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

  // If the dynamic offset is set, and the address is a VGPR, we cannot
  // move the dynamic offset to the mem op.
  if (dynamicOffset || isVGPRAddr)
    return success(changed);

  Value dynOff = ptrAdd.getDynamicOffset();

  // If there's no dynamic offset and the constant offset is already merged,
  // we're done.
  if (!dynOff && newConstOff <= 0)
    return success(changed);

  // Create the constant offset.
  Value cst =
      getI32Constant(rewriter, op->getLoc(), static_cast<int32_t>(newConstOff));

  // If we don't have a dynamic offset, use the constant offset as the dynamic
  // offset and set the constant offset to 0.
  if (!dynOff) {
    Value dst = createAllocation(rewriter, op->getLoc(),
                                 getVGPR(rewriter.getContext(), 1));
    dynOff = lsir::MovOp::create(rewriter, op->getLoc(), dst, cst).getDstRes();
    newConstOff = 0;
  }

  // If we have a non-trivial constant offset, add it to the dynamic offset to
  // avoid i64 additions.
  if (newConstOff > 0) {
    Value dst = createAllocation(rewriter, op->getLoc(),
                                 getVGPR(rewriter.getContext(), 1));
    dynOff = lsir::AddIOp::create(rewriter, op->getLoc(),
                                  TypeAttr::get(rewriter.getI32Type()), dst,
                                  dynOff, cst)
                 .getDstRes();
    newConstOff = 0;
  }
  dynamicOffsetMutable().assign(dynOff);
  newDynOff = nullptr;
  changed = true;

  return success(changed);
}

/// Optimize memory operations whose address/voffset is produced by lsir.addi
/// with a constant operand. Folds the constant into the instruction's
/// constant_offset field.
///
/// Applies to:
/// - DS (LDS) ops: folds into 16-bit unsigned offset (max 65535)
/// - Buffer (MUBUF) ops: folds into 12-bit unsigned offset (max 4095)
///
/// DS pattern (folds addr):
///   %addr = lsir.addi i32 %dst, %base, %const
///   amdgcn.store ds_write_b64 data %data addr %addr offset c(%c0)
/// ->
///   amdgcn.store ds_write_b64 data %data addr %base offset c(%const)
///
/// Buffer pattern (folds voffset):
///   %voff = lsir.addi i32 %dst, %base, %const
///   amdgcn.load buffer_load_dwordx4 ... offset u(%soff) + d(%voff) + c(%c0)
/// ->
///   amdgcn.load buffer_load_dwordx4 ... offset u(%soff) + d(%base) + c(%const)
///
/// This eliminates a v_add_u32 per memory operation by absorbing the constant
/// offset into the instruction encoding.
/// Get the foldable operand and its mutable accessor for addi offset folding.
/// For DS/global ops: the addr operand. For buffer ops: the dynamic_offset
/// (voffset).
struct FoldableOperand {
  Value value;
  std::function<MutableOperandRange()> getMutable;
};
static std::optional<FoldableOperand>
getFoldableOperand(Operation *op, const InstMetadata *md, Value addr,
                   GetOperandsFn addrMutable) {
  // DS ops: fold from the addr operand (32-bit LDS address).
  if (md->hasProp(InstProp::Dsmem))
    return FoldableOperand{addr, addrMutable};

  // Buffer ops: fold from the dynamic_offset (voffset), not addr (rsrc).
  if (md->hasProp(InstProp::Buffer)) {
    if (auto loadOp = dyn_cast<amdgcn::LoadOp>(op)) {
      Value dynOff = loadOp.getDynamicOffset();
      if (!dynOff)
        return std::nullopt;
      return FoldableOperand{dynOff, [loadOp]() mutable {
                               return loadOp.getDynamicOffsetMutable();
                             }};
    }
    if (auto storeOp = dyn_cast<amdgcn::StoreOp>(op)) {
      Value dynOff = storeOp.getDynamicOffset();
      if (!dynOff)
        return std::nullopt;
      return FoldableOperand{dynOff, [storeOp]() mutable {
                               return storeOp.getDynamicOffsetMutable();
                             }};
    }
  }
  return std::nullopt;
}

static LogicalResult optimizeAddiOffsets(Operation *op, const InstMetadata *md,
                                         Value addr, Value constantOffset,
                                         GetOperandsFn addrMutable,
                                         GetOperandsFn constantOffsetMutable,
                                         PatternRewriter &rewriter) {
  // Apply to DS or buffer memory operations.
  // Global ops use 64-bit addresses via ptr_add, handled by
  // optimizeMemOpOffsets.
  if (!md || !md->hasAnyProps({InstProp::Dsmem, InstProp::Buffer}))
    return rewriter.notifyMatchFailure(op, "not a DS or buffer memory op");

  auto foldable = getFoldableOperand(op, md, addr, addrMutable);
  if (!foldable)
    return rewriter.notifyMatchFailure(op, "no foldable operand");

  // Check if the foldable operand is produced by lsir.addi.
  auto addi = foldable->value.getDefiningOp<lsir::AddIOp>();
  if (!addi)
    return rewriter.notifyMatchFailure(op, "operand not produced by lsir.addi");

  // Check if one of the addi operands is a constant i32.
  // Guard: only call getConstant on integer-typed values (not register types).
  Value lhs = addi.getLhs();
  Value rhs = addi.getRhs();
  Value base = nullptr;
  int64_t addiConst = 0;

  auto tryGetConst = [](Value v) -> std::optional<int32_t> {
    if (!isa<IntegerType>(v.getType()))
      return std::nullopt;
    return ValueOrI32::getConstant(v);
  };

  if (auto cst = tryGetConst(rhs)) {
    base = lhs;
    addiConst = *cst;
  } else if (auto cst = tryGetConst(lhs)) {
    base = rhs;
    addiConst = *cst;
  } else {
    return rewriter.notifyMatchFailure(op, "neither addi operand is constant");
  }

  // Get the existing constant offset from the mem op.
  int32_t memOpOff = 0;
  if (constantOffset) {
    std::optional<int32_t> c = ValueOrI32::getConstant(constantOffset);
    if (!c)
      return rewriter.notifyMatchFailure(op, "expected constant offset");
    memOpOff = *c;
  }

  // Compute merged offset with instruction-type-specific limit.
  int64_t maxOff = getMaxConstOffset(md);
  int64_t mergedOff = addiConst + memOpOff;
  if (mergedOff < 0 || mergedOff > maxOff)
    return rewriter.notifyMatchFailure(op, "merged offset out of range");

  // Fold: use the non-constant addi operand, put constant in offset.
  rewriter.modifyOpInPlace(op, [&] {
    foldable->getMutable().assign(base);
    constantOffsetMutable().assign(getI32Constant(
        rewriter, op->getLoc(), static_cast<int32_t>(mergedOff)));
  });
  return success();
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
  // Try ptr_add pattern first, then lsir.addi pattern for DS/buffer ops.
  if (succeeded(optimizeMemOpOffsets(
          op.getOperation(), md, op.getAddr(), op.getConstantOffset(),
          op.getDynamicOffset(), getAddr, getConstantOffset, getDynamicOffset,
          rewriter)))
    return success();
  return optimizeAddiOffsets(op.getOperation(), md, op.getAddr(),
                             op.getConstantOffset(), getAddr, getConstantOffset,
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
  // Try ptr_add pattern first, then lsir.addi pattern for DS/buffer ops.
  if (succeeded(optimizeMemOpOffsets(
          op.getOperation(), md, op.getAddr(), op.getConstantOffset(),
          op.getDynamicOffset(), getAddr, getConstantOffset, getDynamicOffset,
          rewriter)))
    return success();
  return optimizeAddiOffsets(op.getOperation(), md, op.getAddr(),
                             op.getConstantOffset(), getAddr, getConstantOffset,
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

  if (failed(applyPatternsGreedily(
          op, FrozenRewritePatternSet(std::move(patterns)),
          GreedyRewriteConfig().setUseTopDownTraversal(true)))) {
    return signalPassFailure();
  }
}
