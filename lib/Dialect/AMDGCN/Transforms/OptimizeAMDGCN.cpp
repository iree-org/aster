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

/// Optimize memory operations whose address/voffset is produced by lsir.addi
/// with a constant operand. Folds the constant into the instruction's
/// constant_offset field.
///
/// DS pattern (folds addr):
///   %addr = lsir.addi i32 %dst, %base, %const
///   amdgcn.ds_write_b64 ins(%addr, %data) args(%c0)
/// ->
///   amdgcn.ds_write_b64 ins(%base, %data) args(%const)
///
/// Buffer pattern (folds voffset):
///   %voff = lsir.addi i32 %dst, %base, %const
///   amdgcn.buffer_load_dword ... off_or_idx = %voff ... args(%c0)
/// ->
///   amdgcn.buffer_load_dword ... off_or_idx = %base ... args(%const)
static LogicalResult optimizeAddiOffsets(Operation *op, Operand foldableOperand,
                                         Operand constOffset,
                                         int64_t maxConstOff,
                                         PatternRewriter &rewriter) {
  Value foldableValue = foldableOperand.getValue();
  if (!foldableValue)
    return rewriter.notifyMatchFailure(op, "no foldable operand value");

  auto addi = foldableValue.getDefiningOp<lsir::AddIOp>();
  if (!addi)
    return rewriter.notifyMatchFailure(op, "operand not produced by lsir.addi");

  // Check if one of the addi operands is a constant i32.
  Value lhs = addi.getLhs();
  Value rhs = addi.getRhs();
  Value base = nullptr;
  int64_t addiConst = 0;

  auto tryGetConst = [](Value v) -> std::optional<int32_t> {
    if (!isa<IntegerType>(v.getType()))
      return std::nullopt;
    return ValueOrI32::getConstant(v);
  };

  std::optional<int32_t> cst = tryGetConst(rhs);
  if (cst) {
    base = lhs;
    addiConst = *cst;
  } else {
    cst = tryGetConst(lhs);
    if (!cst)
      return rewriter.notifyMatchFailure(op,
                                         "neither addi operand is constant");
    base = rhs;
    addiConst = *cst;
  }

  // Get the existing constant offset from the mem op.
  std::optional<int32_t> memOpOffOpt = getConstOffsetValue(constOffset);
  if (!memOpOffOpt)
    return rewriter.notifyMatchFailure(op, "expected constant offset");

  int64_t mergedOff = addiConst + *memOpOffOpt;
  if (mergedOff < 0 || mergedOff > maxConstOff)
    return rewriter.notifyMatchFailure(op, "merged offset out of range");

  rewriter.modifyOpInPlace(op, [&] {
    foldableOperand->set(base);
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
