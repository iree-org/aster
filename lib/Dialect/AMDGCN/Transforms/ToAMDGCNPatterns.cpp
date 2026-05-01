//===- ToAMDGCNPatterns.cpp -----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert to AMDGCN patterns to more complex ops that are too complex for PDLL.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/ValueOrConst.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <utility>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Create an SCC alloca value for use as scc_dst or scc_src operand.
static Value createSCCAlloca(OpBuilder &builder, Location loc) {
  return AllocaOp::create(builder, loc,
                          SCCType::get(builder.getContext(), Register(0)));
}

/// Helper to create an SOP2Out2In2 instruction and return the dst result.
/// These instructions have 2 outs (dst0 + scc_dst) and 2 ins (src0, src1).
template <typename OpTy>
static Value createSOP2Out2In2(OpBuilder &builder, Location loc, Value dst,
                               Value src0, Value src1) {
  Value sccDst = createSCCAlloca(builder, loc);
  return OpTy::create(builder, loc, dst, sccDst, src0, src1).getDst0Res();
}

/// Helper to create an SOP2Out2In3 instruction and return the dst result.
/// These instructions have 2 outs (dst0 + scc_dst) and 3 ins (src0, src1,
/// scc_src). The scc_src alloca is created internally to represent the SCC
/// carry input.
template <typename OpTy>
static Value createSOP2Out2In3(OpBuilder &builder, Location loc, Value dst,
                               Value src0, Value src1) {
  Value sccDst = createSCCAlloca(builder, loc);
  Value sccSrc = createSCCAlloca(builder, loc);
  return OpTy::create(builder, loc, dst, sccDst, src0, src1, sccSrc)
      .getDst0Res();
}

namespace {
//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

struct AddFOpPattern : public OpRewritePattern<lsir::AddFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AddFOp op,
                                PatternRewriter &rewriter) const override;
};

struct AddIOpPattern : public OpRewritePattern<lsir::AddIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AddIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

struct AllocaOpPattern : public OpRewritePattern<lsir::AllocaOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AllocaOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AssumeNoaliasOpPattern
//===----------------------------------------------------------------------===//

struct AssumeNoaliasOpPattern : public OpRewritePattern<lsir::AssumeNoaliasOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AssumeNoaliasOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AndIOpPattern
//===----------------------------------------------------------------------===//

struct AndIOpPattern : public OpRewritePattern<lsir::AndIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::AndIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// KernelOpPattern
//===----------------------------------------------------------------------===//

struct KernelOpPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using Base::Base;
  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//

struct LoadOpPattern : public OpRewritePattern<lsir::LoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MulIOpPattern
//===----------------------------------------------------------------------===//

struct MaximumFOpPattern : public OpRewritePattern<lsir::MaximumFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MaximumFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MinimumFOpPattern : public OpRewritePattern<lsir::MinimumFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MinimumFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulFOpPattern : public OpRewritePattern<lsir::MulFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulFOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulIOpPattern : public OpRewritePattern<lsir::MulIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulIOp op,
                                PatternRewriter &rewriter) const override;
};

struct MulHiSIOpPattern : public OpRewritePattern<lsir::MulHiSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulHiSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MovOpPattern
//===----------------------------------------------------------------------===//

struct MovOpPattern : public OpRewritePattern<lsir::MovOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MovOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// OrIOpPattern
//===----------------------------------------------------------------------===//

struct OrIOpPattern : public OpRewritePattern<lsir::OrIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::OrIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// XOrIOpPattern
//===----------------------------------------------------------------------===//

struct XOrIOpPattern : public OpRewritePattern<lsir::XOrIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::XOrIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegCastOpPattern
//===----------------------------------------------------------------------===//

struct RegCastOpPattern : public OpRewritePattern<lsir::RegCastOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::RegCastOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ReturnOpPattern
//===----------------------------------------------------------------------===//

struct ReturnOpPattern : public OpRewritePattern<func::ReturnOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShLIOpPattern
//===----------------------------------------------------------------------===//

struct ShLIOpPattern : public OpRewritePattern<lsir::ShLIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShLIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ExtSIOpPattern
//===----------------------------------------------------------------------===//

struct ExtSIOpPattern : public OpRewritePattern<lsir::ExtSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ExtSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ExtUIOpPattern
//===----------------------------------------------------------------------===//

struct ExtUIOpPattern : public OpRewritePattern<lsir::ExtUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ExtUIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShRSIOpPattern
//===----------------------------------------------------------------------===//

struct ShRSIOpPattern : public OpRewritePattern<lsir::ShRSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShRSIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ShRUIOpPattern
//===----------------------------------------------------------------------===//

struct ShRUIOpPattern : public OpRewritePattern<lsir::ShRUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::ShRUIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//

struct StoreOpPattern : public OpRewritePattern<lsir::StoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// SubIOpPattern
//===----------------------------------------------------------------------===//

struct SubFOpPattern : public OpRewritePattern<lsir::SubFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SubFOp op,
                                PatternRewriter &rewriter) const override;
};

struct SubIOpPattern : public OpRewritePattern<lsir::SubIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::SubIOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TimingStartOpPattern
//===----------------------------------------------------------------------===//

struct TimingStartOpPattern : public OpRewritePattern<lsir::TimingStartOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TimingStartOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TimingStopOpPattern
//===----------------------------------------------------------------------===//

struct TimingStopOpPattern : public OpRewritePattern<lsir::TimingStopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::TimingStopOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// WaitOpPattern
//===----------------------------------------------------------------------===//

struct WaitOpPattern : public OpRewritePattern<lsir::WaitOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::WaitOp op,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

struct PtrAddOpPattern : public OpRewritePattern<PtrAddOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(PtrAddOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Check if the given width is contained in the list of widths.
static bool hasWidth(unsigned width, ArrayRef<unsigned> widths) {
  return llvm::is_contained(widths, width);
}

/// Check if the given operand kind is contained in the list of kinds.
static bool isOperand(OperandKind value, ArrayRef<OperandKind> kinds) {
  return llvm::is_contained(kinds, value);
}

/// Check if the given type is a valid operand of the given kind and size.
static bool isValidOperand(Type type, ArrayRef<OperandKind> kind,
                           int16_t numWords) {
  OperandKind operandKind = getOperandKind(type);
  if (auto rT = dyn_cast<RegisterTypeInterface>(type)) {
    if (!llvm::is_contained(kind, operandKind))
      return false;
    return rT.getAsRange().size() == numWords;
  }
  return llvm::is_contained(kind, operandKind);
}
static bool isValidOperand(Type type, OperandKind kind, int16_t numWords) {
  return (kind == OperandKind::SGPR &&
          isValidOperand(type, {OperandKind::SGPR, OperandKind::IntImm},
                         numWords)) ||
         (kind == OperandKind::VGPR &&
          isValidOperand(
              type, {OperandKind::SGPR, OperandKind::VGPR, OperandKind::IntImm},
              numWords));
}

/// Helper function to get element from range or default value.
static Value getElemOr(ValueRange range, int32_t i, Value value) {
  if (range.empty())
    return value;
  return range[i];
}

/// Create an i32 constant value.
static Value getI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getIntegerAttr(builder.getI32Type(), value));
}

// Create a new-style VOP instruction with 1 output and 2 inputs.
template <typename OpT>
static Value createNewVOP(OpBuilder &builder, Location loc, Value dst,
                          Value src0, Value src1) {
  return OpT::create(builder, loc, dst, src0, src1).getDst0Res();
}

/// If the destination is SGPR but any source operand is VGPR (mixed uniform/
/// dynamic from assume_uniform), override `kind` to VGPR and set
/// `sgprDestVgprSrc` so the caller can copy the VGPR result back to SGPR.
static void handleMixedSgprVgprDest(OperandKind &kind,
                                    RegisterTypeInterface &oTy, Value lhs,
                                    Value rhs, bool &sgprDestVgprSrc,
                                    MLIRContext *ctx) {
  sgprDestVgprSrc = false;
  if (kind != OperandKind::SGPR)
    return;
  OperandKind lhsK = getOperandKind(lhs.getType());
  OperandKind rhsK = getOperandKind(rhs.getType());
  if (lhsK != OperandKind::VGPR && rhsK != OperandKind::VGPR)
    return;
  sgprDestVgprSrc = true;
  kind = OperandKind::VGPR;
  // Derive VGPR range size from original SGPR type to handle 32- and 64-bit.
  int16_t rangeSize = oTy.getAsRange().size();
  oTy = cast<RegisterTypeInterface>(getVGPR(ctx, rangeSize));
}

/// After a VGPR-path computation for a mixed SGPR dest, copy the VGPR result
/// to the SGPR destination and return the SGPR value.
static Value finishMixedSgprVgprDest(PatternRewriter &rewriter, Location loc,
                                     Value dst, Value vgprResult,
                                     bool sgprDestVgprSrc) {
  if (!sgprDestVgprSrc)
    return vgprResult;
  return lsir::CopyOp::create(rewriter, loc, dst, vgprResult).getTargetRes();
}

/// Check validity of an AMDGCN arith op.
static LogicalResult checkAIOp(Operation *op, PatternRewriter &rewriter,
                               OperandKind kind, Value lhs, Value rhs,
                               RegisterTypeInterface oTy, unsigned width,
                               OperandKind &lhsKind, OperandKind &rhsKind,
                               ArrayRef<unsigned> sgprWidths,
                               ArrayRef<unsigned> vgprWidths) {
  // Check that the output type is an AMDGCN register type
  if (!isAMDReg(oTy)) {
    return rewriter.notifyMatchFailure(
        op, "operand type is not an AMDGCN register type");
  }

  // AGPRs are not supported for arith operations
  if (kind == OperandKind::AGPR)
    return rewriter.notifyMatchFailure(op, "operand type cannot be AGPR");
  int16_t rangeSize = oTy.getAsRange().size();

  if (rangeSize != ((width / 8) + 3) / 4) {
    return rewriter.notifyMatchFailure(
        op, "register range size does not match the operation width");
  }

  // Validate supported widths
  if (kind == OperandKind::SGPR && !hasWidth(width, sgprWidths)) {
    return rewriter.notifyMatchFailure(
        op, "SGPR arith operations only support 32 or 64-bit widths");
  }
  if (kind == OperandKind::VGPR && !hasWidth(width, vgprWidths)) {
    return rewriter.notifyMatchFailure(
        op, "VGPR arith operations only support 16, 32, or 64-bit widths");
  }

  // Validate lhs and rhs operand types
  lhsKind = getOperandKind(lhs.getType());
  if (!isValidOperand(lhs.getType(), kind, rangeSize)) {
    return rewriter.notifyMatchFailure(
        op, "Invalid lhs operand type for arith operation");
  }
  rhsKind = getOperandKind(rhs.getType());
  if (!isValidOperand(rhs.getType(), kind, rangeSize)) {
    return rewriter.notifyMatchFailure(
        op, "Invalid rhs operand type for arith operation");
  }

  // Both operands shouldn't be immediates
  if (lhsKind == OperandKind::IntImm && rhsKind == OperandKind::IntImm) {
    return rewriter.notifyMatchFailure(
        op, "Expected at least one non-immediate operand for add operation");
  }
  return success();
}

static MLIRContext *getCtx(PatternRewriter &rewriter) {
  return rewriter.getContext();
}

//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult AddIOpPattern::matchAndRewrite(lsir::AddIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but a source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind/oTy for validation.
  bool sgprDestVgprSrc = false;
  handleMixedSgprVgprDest(kind, oTy, lhs, rhs, sgprDestVgprSrc,
                          op.getContext());

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // At this point, operands are valid - create the appropriate add op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAddU32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo = createSOP2Out2In2<SAddU32>(
        rewriter, loc, getElemOr(dstR, 0, dst), getElemOr(lhsR, 0, lhs),
        getElemOr(rhsR, 0, rhs));
    Value hi = createSOP2Out2In3<SAddcU32>(
        rewriter, loc, getElemOr(dstR, 1, dst), getElemOr(lhsR, 1, lhs),
        getElemOr(rhsR, 1, rhs));
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    int16_t rangeSize = oTy.getAsRange().size();
    actualDst = createAllocation(rewriter, loc,
                                 getVGPR(rewriter.getContext(), rangeSize));
  }

  // Handle the VGPR case
  Value result;
  if (width <= 16) {
    result = createNewVOP<VAddU16>(rewriter, loc, actualDst, lhs, rhs);
  } else if (width <= 32) {
    result = createNewVOP<VAddU32>(rewriter, loc, actualDst, lhs, rhs);
  } else {
    // 64-bit VGPR add
    result = VLshlAddU64::create(rewriter, loc, actualDst, lhs,
                                 getI32Constant(rewriter, loc, 0), rhs)
                 .getDst0Res();
  }

  // Copy VGPR result back to SGPR destination.
  result = finishMixedSgprVgprDest(rewriter, loc, dst, result, sgprDestVgprSrc);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// Float binary op patterns (AddF, SubF, MulF, MaximumF, MinimumF)
//===----------------------------------------------------------------------===//

/// Generic lowering for lsir binary float ops -> new VOP instructions.
/// Float ops are always VGPR (no SGPR float arithmetic on AMD GPUs).
template <typename LsirOp, typename VOp32, typename VOp16>
static LogicalResult lowerBinaryFloatOp(LsirOp op, PatternRewriter &rewriter) {
  RegisterTypeInterface oTy = op.getDst().getType();
  OperandKind kind = getOperandKind(oTy);
  if (kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "float ops require VGPR dest");

  unsigned width = op.getSemantics().getWidth();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Location loc = op.getLoc();

  OperandKind rhsKind = getOperandKind(rhs.getType());

  // Commute if rhs is immediate/SGPR (VOP2 src0 can be SGPR, src1 must be
  // VGPR).
  OperandKind lhsKind = getOperandKind(lhs.getType());
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  if (width <= 16) {
    Value result = createNewVOP<VOp16>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createNewVOP<VOp32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "64-bit float VOP not supported");
}

LogicalResult AddFOpPattern::matchAndRewrite(lsir::AddFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::AddFOp, VAddF32, VAddF16>(op, rewriter);
}

LogicalResult SubFOpPattern::matchAndRewrite(lsir::SubFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::SubFOp, VSubF32, VSubF32>(op, rewriter);
}

LogicalResult MulFOpPattern::matchAndRewrite(lsir::MulFOp op,
                                             PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MulFOp, VMulF32, VMulF16>(op, rewriter);
}

LogicalResult
MaximumFOpPattern::matchAndRewrite(lsir::MaximumFOp op,
                                   PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MaximumFOp, VMaxF32, VMaxF32>(op, rewriter);
}

LogicalResult
MinimumFOpPattern::matchAndRewrite(lsir::MinimumFOp op,
                                   PatternRewriter &rewriter) const {
  return lowerBinaryFloatOp<lsir::MinimumFOp, VMinF32, VMinF32>(op, rewriter);
}

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(lsir::AllocaOp op,
                                 PatternRewriter &rewriter) const {
  // Check that the output type is an AMDGCN register type
  if (!isAMDReg(op.getType())) {
    return rewriter.notifyMatchFailure(
        op, "operand type is not an AMDGCN register type");
  }
  rewriter.replaceOp(op, createAllocation(rewriter, op.getLoc(), op.getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// AssumeNoaliasOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AssumeNoaliasOpPattern::matchAndRewrite(lsir::AssumeNoaliasOp op,
                                        PatternRewriter &rewriter) const {
  // If we still have AssumeNoAlias at this point, just forward the operands.
  // This op is meant to be used in analyses before lowering to improve alias
  // analysis.
  rewriter.replaceOp(op, op.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// AndIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult AndIOpPattern::matchAndRewrite(lsir::AndIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {32})))
    return failure();

  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // At this point, operands are valid - create the appropriate and op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAndB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SAndB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  // Handle the VGPR case
  Value result = createNewVOP<VAndB32>(rewriter, loc, dst, lhs, rhs);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOpPattern
//===----------------------------------------------------------------------===//

/// Populate the kernel argument list.
static void addKerArg(SmallVectorImpl<KernelArgAttrInterface> &kerArgs,
                      Type hTy, Type gTy, int32_t size, int32_t align) {
  if (auto pTy = dyn_cast<ptr::PtrType>(hTy)) {
    auto arg = BufferArgAttr::get(gTy.getContext(), AddressSpaceKind::Global,
                                  AccessKind::ReadWrite,
                                  KernelArgumentFlags::None, "", hTy);
    kerArgs.push_back(arg);
    return;
  }
  auto arg = ByValueArgAttr::get(gTy.getContext(), size, align, "", hTy);
  kerArgs.push_back(arg);
}

/// Add hidden kernel arguments.
static void hiddenArgs(SmallVectorImpl<KernelArgAttrInterface> &kerArgs,
                       MLIRContext *ctx) {
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::X));
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::Y));
  kerArgs.push_back(BlockDimArgAttr::get(ctx, Dim::Z));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::X));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::Y));
  kerArgs.push_back(GridDimArgAttr::get(ctx, Dim::Z));
}

LogicalResult
KernelOpPattern::matchAndRewrite(FunctionOpInterface op,
                                 PatternRewriter &rewriter) const {
  if (isa<KernelOp>(op.getOperation()))
    return rewriter.notifyMatchFailure(op, "already a kernel operation");
  auto gFn = dyn_cast<GPUFuncInterface>(op.getOperation());
  // Check this is a GPU kernel function
  if (!gFn || !gFn.isGPUKernel())
    return rewriter.notifyMatchFailure(op, "not a GPU kernel function");

  auto gpuAbiTy = cast<FunctionType>(op.getFunctionType());

  // Get the host ABI information
  auto [type, typeSizes, alignment] = gFn.getHostABI();
  if (!type || type.getNumInputs() != op.getNumArguments() ||
      type.getNumResults() != op.getNumResults()) {
    return rewriter.notifyMatchFailure(op, "invalid host ABI function type");
  }
  if (typeSizes.size() != type.getNumInputs())
    return rewriter.notifyMatchFailure(op, "invalid host ABI size array");
  if (alignment.size() != type.getNumInputs())
    return rewriter.notifyMatchFailure(op, "invalid host ABI alignment array");
  if (!llvm::all_of(gpuAbiTy.getInputs(), llvm::IsaPred<SGPRType, SGPRType>))
    return rewriter.notifyMatchFailure(op, "expected all inputs to be SGPRs");

  // Set Metadata attributes
  int32_t smemSize = gFn.getSharedMemorySize();
  SmallVector<KernelArgAttrInterface> kerArgs;
  for (auto [hTy, dTy, sz, align] :
       llvm::zip(cast<FunctionType>(op.getFunctionType()).getInputs(),
                 type.getInputs(), typeSizes, alignment)) {
    addKerArg(kerArgs, hTy, dTy, sz, align);
  }
  hiddenArgs(kerArgs, op.getContext());

  // Create the KernelOp
  auto kOp = amdgcn::KernelOp::create(
      rewriter, op.getLoc(), op.getName(), kerArgs, smemSize,
      /*private_memory_size=*/0, /*enable_private_segment_buffer=*/false,
      /*enable_dispatch_ptr=*/false,
      /*enable_kernarg_segment_ptr=*/!kerArgs.empty());
  rewriter.inlineRegionBefore(op.getFunctionBody(), kOp.getBodyRegion(),
                              kOp.getBodyRegion().end());
  Block *entry = &kOp.getBodyRegion().front();
  // Replace arguments with LoadArgOps
  rewriter.setInsertionPointToStart(entry);
  int64_t numArgs = op.getNumArguments();
  for (auto [i, arg] : llvm::enumerate(llvm::reverse(entry->getArguments()))) {
    int64_t idx = numArgs - i - 1;
    assert(idx >= 0 && idx < numArgs && "invalid argument index");
    if (arg.use_empty())
      continue;
    Value rA = LoadArgOp::create(rewriter, op.getLoc(), arg.getType(), idx);
    rewriter.replaceAllUsesWith(arg, rA);
  }
  entry->eraseArguments(0, numArgs);

  // Replace the function op with the kernel op
  rewriter.replaceOp(op, kOp);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOpPattern
//===----------------------------------------------------------------------===//

LogicalResult LoadOpPattern::matchAndRewrite(lsir::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto memSpace = cast<amdgcn::AddressSpaceAttr>(op.getMemorySpace());
  if (!memSpace) {
    return rewriter.notifyMatchFailure(
        op, "expected AMDGCN address space attribute for load operation");
  }

  // Check dependencies
  if (op.getDependencies().size() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "load operation with dependencies are not supported by this pattern");
  }
  if (!op.getOutDependency().use_empty()) {
    return rewriter.notifyMatchFailure(
        op, "can't handle load operation with out dependency in this pattern");
  }

  // Check memory space
  AddressSpaceKind space = memSpace.getSpace();
  if (!isAddressSpaceOf(space,
                        {AddressSpaceKind::Global, AddressSpaceKind::Local})) {
    return rewriter.notifyMatchFailure(
        op,
        "only global and local memory spaces are supported by this pattern");
  }

  // Get constant offset
  int32_t off = 0;
  if (std::optional<int32_t> constOff =
          ValueOrI32::getConstant(op.getConstOffset())) {
    off = *constOff;
  } else {
    return rewriter.notifyMatchFailure(
        op, "only constant offsets are supported by this pattern");
  }

  Location loc = op.getLoc();
  TypedValue<RegisterTypeInterface> dst = op.getDst();
  TypedValue<RegisterTypeInterface> addr = op.getAddr();
  Value offset = op.getOffset();
  RegisterTypeInterface addrTy = addr.getType();
  RegisterTypeInterface resTy = dst.getType();
  Value result;

  // Check if the offset is constant and add it to the constant offset.
  if (std::optional<int32_t> constOff = ValueOrI32::getConstant(offset)) {
    off += *constOff;
    offset = nullptr;
  }

  // Number of 32-bit words to load
  int16_t numWords = resTy.getAsRange().size();
  if (space == AddressSpaceKind::Local) {
    if (!isVGPR(addrTy, 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected VGPR address for load from shared memory space");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op,
                                         "only constant offsets are supported "
                                         "for load from shared memory space");
    }
    offset = getI32Constant(rewriter, loc, off);
    switch (numWords) {
    case 1:
      result =
          DS_READ_B32::create(rewriter, loc, dst, addr, offset).getDestRes();
      break;
    case 2:
      result =
          DS_READ_B64::create(rewriter, loc, dst, addr, offset).getDestRes();
      break;
    case 3:
      result =
          DS_READ_B96::create(rewriter, loc, dst, addr, offset).getDestRes();
      break;
    case 4:
      result =
          DS_READ_B128::create(rewriter, loc, dst, addr, offset).getDestRes();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from shared memory space");
    }
    rewriter.replaceAllUsesWith(op.getDstRes(), result);
    rewriter.eraseOp(op);
    return success();
  }
  Value cOff;
  if (off > 0) {
    cOff = getI32Constant(rewriter, loc, off);
  }

  // Handle a SMEM load
  bool addrIsSGPR = isSGPR(addrTy, 2);
  if (addrIsSGPR && (!offset || isSGPR(offset.getType(), 1))) {
    if (offset) {
      return rewriter.notifyMatchFailure(op, "nyi: SGPR offset");
    }
    switch (numWords) {
    case 1:
      result = S_LOAD_DWORD::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 2:
      result = S_LOAD_DWORDX2::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 4:
      result = S_LOAD_DWORDX4::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 8:
      result = S_LOAD_DWORDX8::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    case 16:
      result = S_LOAD_DWORDX16::create(rewriter, loc, dst, addr, nullptr, cOff)
                   .getDestRes();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from shared memory space");
    }
    rewriter.replaceAllUsesWith(op.getDstRes(), result);
    rewriter.eraseOp(op);
    return success();
  }

  // Handle a VMEM load
  bool addrIsVGPR = isVGPR(addrTy, 2);
  if (!addrIsVGPR && !addrIsSGPR) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR or SGPR address for load from global memory space");
  }
  if (addrIsVGPR && offset) {
    return rewriter.notifyMatchFailure(
        op, "expected no offset or SGPR address for load");
  }
  if (addrIsSGPR && offset && !isVGPR(offset.getType(), 1)) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR offset for load from global memory space");
  }
  if (true)
    switch (numWords) {
    case 1:
      result = GLOBAL_LOAD_DWORD::create(rewriter, loc, dst, addr, offset, cOff)
                   .getDestRes();
      break;
    case 2:
      result =
          GLOBAL_LOAD_DWORDX2::create(rewriter, loc, dst, addr, offset, cOff)
              .getDestRes();
      break;
    case 3:
      result =
          GLOBAL_LOAD_DWORDX3::create(rewriter, loc, dst, addr, offset, cOff)
              .getDestRes();
      break;
    case 4:
      result =
          GLOBAL_LOAD_DWORDX4::create(rewriter, loc, dst, addr, offset, cOff)
              .getDestRes();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from global memory space");
    }
  rewriter.replaceAllUsesWith(op.getDstRes(), result);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// MulIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult MulIOpPattern::matchAndRewrite(lsir::MulIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but a source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind/oTy for validation.
  bool sgprDestVgprSrc = false;
  handleMixedSgprVgprDest(kind, oTy, lhs, rhs, sgprDestVgprSrc,
                          op.getContext());

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // If lhs is a constant that doesn't fit in 6 bits, move it to a VGPR.
  // TODO: this is just a very quick and approximate fix, we should have a
  // general solution.
  if (kind == OperandKind::VGPR && lhsKind == OperandKind::IntImm) {
    APInt constVal;
    if (matchPattern(lhs, m_ConstantInt(&constVal)) &&
        !constVal.isSignedIntN(6)) {
      Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
      lhs = VMovB32::create(rewriter, loc, vgpr, lhs).getDst0Res();
      lhsKind = OperandKind::VGPR;
    }
  }

  // At this point, operands are valid - create the appropriate mul op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = SMulI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
      rewriter.replaceOp(op, result);
      return success();
    }

    // 64-bit SGPR multiplication
    Value lLo = getElemOr(lhsR, 0, lhs);
    Value lHi = getElemOr(lhsR, 1, lhs);
    Value rLo = getElemOr(rhsR, 0, rhs);
    Value rHi = getElemOr(rhsR, 1, rhs);
    Value dLo = getElemOr(dstR, 0, dst);
    Value dHi = getElemOr(dstR, 1, dst);
    Value t0 = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter)));
    Value t1 = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter)));

    dHi = SMulI32::create(rewriter, loc, dHi, rLo, lHi).getDst0Res();
    t0 = SMulHiU32::create(rewriter, loc, t0, rLo, lLo).getDst0Res();
    t1 = SMulI32::create(rewriter, loc, t1, rHi, lLo).getDst0Res();
    dHi = createSOP2Out2In2<SAddI32>(rewriter, loc, dHi, t0, dHi);
    dLo = SMulI32::create(rewriter, loc, dLo, rLo, lLo).getDst0Res();
    dHi = createSOP2Out2In2<SAddI32>(rewriter, loc, dHi, dHi, t1);

    // Combine low and high parts
    Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dLo, dHi});
    rewriter.replaceOp(op, result);
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    int16_t rangeSize = oTy.getAsRange().size();
    actualDst = createAllocation(rewriter, loc,
                                 getVGPR(rewriter.getContext(), rangeSize));
  }

  // Handle the VGPR case
  Value result;
  if (width <= 16) {
    result = createNewVOP<VMulLoU16>(rewriter, loc, actualDst, lhs, rhs);
  } else if (width <= 32) {
    if (lhsKind == OperandKind::IntImm) {
      APInt constVal;
      if (matchPattern(lhs, m_ConstantInt(&constVal))) {
        Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
        lhs = VMovB32::create(rewriter, loc, vgpr, lhs).getDst0Res();
        lhsKind = OperandKind::VGPR;
      }
    }
    result = VMulLoU32::create(rewriter, loc, actualDst, lhs, rhs).getDst0Res();
  } else {
    // 64-bit VGPR multiplication
    Value lLo = getElemOr(lhsR, 0, lhs);
    Value lHi = getElemOr(lhsR, 1, lhs);
    Value rLo = getElemOr(rhsR, 0, rhs);
    Value rHi = getElemOr(rhsR, 1, rhs);

    // Allocate temporaries
    Value t0 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
    Value t1 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
    Value carry = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter), 2));
    t0 = VMulLoU32::create(rewriter, loc, t0, rHi, lLo).getDst0Res();
    t1 = VMulLoU32::create(rewriter, loc, t1, rLo, lHi).getDst0Res();
    Value zero = getI32Constant(rewriter, loc, 0);
    ValueRange dT0 = splitRange(
        rewriter, loc,
        VMadU64U32::create(rewriter, loc, actualDst, carry, rLo, lLo, zero)
            .getDst0Res());
    Value t3 =
        VAdd3U32::create(rewriter, loc, dT0[1], dT0[1], t1, t0).getDst0Res();
    result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dT0[0], t3});
  }

  // Copy VGPR result back to SGPR destination.
  result = finishMixedSgprVgprDest(rewriter, loc, dst, result, sgprDestVgprSrc);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// MulHiSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
MulHiSIOpPattern::matchAndRewrite(lsir::MulHiSIOp op,
                                  PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32}, {32})))
    return failure();

  Location loc = op.getLoc();

  if (kind == OperandKind::SGPR) {
    Value result = SMulHiI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }

  // VOP3 does not support literal constants -- move to VGPR first.
  auto movLargeImm = [&](Value &v, OperandKind vKind) {
    if (vKind != OperandKind::IntImm)
      return;
    APInt constVal;
    if (matchPattern(v, m_ConstantInt(&constVal)) &&
        !constVal.isSignedIntN(6)) {
      Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
      v = VMovB32::create(rewriter, loc, vgpr, v).getDst0Res();
    }
  };
  movLargeImm(lhs, lhsKind);
  movLargeImm(rhs, rhsKind);

  Value result = VMulHiI32::create(rewriter, loc, dst, lhs, rhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// OrIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult OrIOpPattern::matchAndRewrite(lsir::OrIOp op,
                                            PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {32})))
    return failure();

  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // At this point, operands are valid - create the appropriate or op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SOrB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SOrB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  Value result = createNewVOP<VOrB32>(rewriter, loc, dst, lhs, rhs);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// XOrIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult XOrIOpPattern::matchAndRewrite(lsir::XOrIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {32})))
    return failure();

  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // At this point, operands are valid - create the appropriate xor op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SXorB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SXorB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsKind, rhsKind);
  }

  // Handle the VGPR case
  Value result = createNewVOP<VXorB32>(rewriter, loc, dst, lhs, rhs);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// MovOpPattern
//===----------------------------------------------------------------------===//

LogicalResult MovOpPattern::matchAndRewrite(lsir::MovOp op,
                                            PatternRewriter &rewriter) const {
  // Only handle the constant case.
  if (!matchPattern(op.getValue(), m_Constant()))
    return rewriter.notifyMatchFailure(op, "only constant mov is supported");

  OperandKind kind = getOperandKind(op.getDst().getType());
  Value res;
  switch (kind) {
  case OperandKind::VGPR:
    res = VMovB32::create(rewriter, op.getLoc(), op.getDst(), op.getValue())
              .getDst0Res();
    break;
  case OperandKind::SGPR:
    res = SMovB32::create(rewriter, op.getLoc(), op.getDst(), op.getValue())
              .getDst0Res();
    break;
  default:
    return rewriter.notifyMatchFailure(op, "unsupported mov register operand");
  }
  rewriter.replaceOp(op, res);
  return success();
}

//===----------------------------------------------------------------------===//
// RegCastOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
RegCastOpPattern::matchAndRewrite(lsir::RegCastOp op,
                                  PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  OperandKind srcKind = getOperandKind(op.getSrc().getType());
  OperandKind tgtKind = getOperandKind(op.getType());
  if (srcKind != OperandKind::SGPR || tgtKind != OperandKind::VGPR) {
    return rewriter.notifyMatchFailure(
        op, "Can only handle SGPR to VGPR conversion");
  }
  if (op.getSrc().getType().getAsRange().size() != 1 ||
      op.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Can only handle single word conversion conversion");
  }

  Value res = VMovB32::create(rewriter, loc,
                              createAllocation(rewriter, loc, op.getType()),
                              op.getSrc())
                  .getDst0Res();
  rewriter.replaceOp(op, res);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(func::ReturnOp op,
                                 PatternRewriter &rewriter) const {
  if (op->getParentOfType<KernelOp>() == nullptr)
    return failure();
  rewriter.replaceOpWithNewOp<EndKernelOp>(op);
  return success();
}

//===----------------------------------------------------------------------===//
// ShLIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShLIOpPattern::matchAndRewrite(lsir::ShLIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // If dest is SGPR but source is VGPR (mixed uniform/dynamic), use VGPR
  // path and copy result back. Override kind for validation.
  bool sgprDestVgprSrc = false;
  OperandKind srcKind = getOperandKind(lhs.getType());
  if (kind == OperandKind::SGPR && srcKind == OperandKind::VGPR) {
    sgprDestVgprSrc = true;
    kind = OperandKind::VGPR;
    oTy = cast<RegisterTypeInterface>(getVGPR(rewriter.getContext(), 1));
  }

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = isOperand(rhsKind, {OperandKind::SGPR, OperandKind::IntImm});

  // Handle the SGPR case (only if all operands fit SGPR path).
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SLshlB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SLshlB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Allocate VGPR temp if we're writing to SGPR from VGPR path.
  Value actualDst = dst;
  if (sgprDestVgprSrc) {
    actualDst =
        createAllocation(rewriter, loc, getVGPR(rewriter.getContext(), 1));
  }

  // Handle the VGPR case
  Value result;
  if (width == 16) {
    // NOTE: Operands are reversed
    result = createNewVOP<VLshlrevB16>(rewriter, loc, actualDst, rhs, lhs);
  } else if (width == 32) {
    // NOTE: Operands are reversed
    result = createNewVOP<VLshlrevB32>(rewriter, loc, actualDst, rhs, lhs);
  } else {
    // NOTE: Operands are reversed
    result =
        VLshlrevB64::create(rewriter, loc, actualDst, rhs, lhs).getDst0Res();
  }

  // Copy VGPR result back to SGPR destination.
  if (sgprDestVgprSrc) {
    result = lsir::CopyOp::create(rewriter, loc, dst, result).getTargetRes();
  }
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRSIOpPattern::matchAndRewrite(lsir::ShRSIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = isOperand(rhsKind, {OperandKind::SGPR, OperandKind::IntImm});

  // Handle the SGPR case
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SAshrI32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SAshrI64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VAshrrevI16>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VAshrrevI32>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result = VAshrrevI64::create(rewriter, loc, dst, rhs, lhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRUIOpPattern::matchAndRewrite(lsir::ShRUIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {32, 64}, {16, 32, 64})))
    return failure();
  Location loc = op.getLoc();
  // Determine whether we need to use VOP3.
  bool isVOp3 = isOperand(rhsKind, {OperandKind::SGPR, OperandKind::IntImm});

  // Handle the SGPR case
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SLshrB32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = createSOP2Out2In2<SLshrB64>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VLshrrevB16>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createNewVOP<VLshrrevB32>(rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result = VLshrrevB64::create(rewriter, loc, dst, rhs, lhs).getDst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ExtSIOpPattern::matchAndRewrite(lsir::ExtSIOp op,
                                              PatternRewriter &rewriter) const {
  unsigned srcWidth = op.getSrcType().getWidth();
  unsigned tgtWidth = op.getTgtType().getWidth();
  if (srcWidth != 32 || tgtWidth != 64)
    return rewriter.notifyMatchFailure(
        op, "only i32 to i64 and u32 to u64 sign extension is supported");

  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value value = op.getValue();
  OperandKind kind = getOperandKind(oTy);

  if (!isAMDReg(oTy))
    return rewriter.notifyMatchFailure(op, "dst must be AMDGCN register type");
  if (kind != OperandKind::SGPR && kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only SGPR and VGPR are supported");
  if (oTy.getAsRange().size() != 2)
    return rewriter.notifyMatchFailure(
        op, "dst must be a 2-register range for i64");

  Location loc = op.getLoc();
  ValueRange dstR = splitRange(rewriter, loc, dst);
  if (dstR.size() != 2)
    return rewriter.notifyMatchFailure(op, "dst must be a splittable range");

  Value dstLo = dstR[0];
  Value dstHi = dstR[1];

  // Sign extension: lo = value, hi = sign_extend(value) = shrsi(value, 31)
  Value shiftAmount = getI32Constant(rewriter, loc, 31);
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value lo = lsir::CopyOp::create(rewriter, loc, dstLo, value).getTargetRes();
  Value hi = lsir::ShRSIOp::create(rewriter, loc, i32Semantics, dstHi, value,
                                   shiftAmount)
                 .getDstRes();

  Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi});
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ExtUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ExtUIOpPattern::matchAndRewrite(lsir::ExtUIOp op,
                                              PatternRewriter &rewriter) const {
  unsigned srcWidth = op.getSrcType().getWidth();
  unsigned tgtWidth = op.getTgtType().getWidth();
  if (srcWidth != 32 || tgtWidth != 64)
    return rewriter.notifyMatchFailure(
        op, "only i32 to i64 and u32 to u64 zero extension is supported");

  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value value = op.getValue();
  OperandKind kind = getOperandKind(oTy);

  if (!isAMDReg(oTy))
    return rewriter.notifyMatchFailure(op, "dst must be AMDGCN register type");
  if (kind != OperandKind::SGPR && kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only SGPR and VGPR are supported");
  if (oTy.getAsRange().size() != 2)
    return rewriter.notifyMatchFailure(
        op, "dst must be a 2-register range for i64");

  Location loc = op.getLoc();
  ValueRange dstR = splitRange(rewriter, loc, dst);
  if (dstR.size() != 2)
    return rewriter.notifyMatchFailure(op, "dst must be a splittable range");

  Value dstLo = dstR[0];
  Value dstHi = dstR[1];

  // Zero extension: mov 0 -> hi, copy value -> lo
  Value zero = getI32Constant(rewriter, loc, 0);
  Value hi = lsir::MovOp::create(rewriter, loc, dstHi, zero).getDstRes();
  Value lo = lsir::CopyOp::create(rewriter, loc, dstLo, value).getTargetRes();

  Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi});
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOpPattern
//===----------------------------------------------------------------------===//

LogicalResult StoreOpPattern::matchAndRewrite(lsir::StoreOp op,
                                              PatternRewriter &rewriter) const {
  auto memSpace = dyn_cast<amdgcn::AddressSpaceAttr>(op.getMemorySpace());
  if (!memSpace) {
    return rewriter.notifyMatchFailure(
        op, "expected AMDGCN address space attribute for store operation");
  }

  // Check dependencies
  if (op.getDependencies().size() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "store operation with dependencies are not supported by this pattern");
  }
  if (!op.getOutDependency().use_empty()) {
    return rewriter.notifyMatchFailure(
        op, "can't handle load operation with out dependency in this pattern");
  }

  // Check memory space
  AddressSpaceKind space = memSpace.getSpace();
  if (!isAddressSpaceOf(space,
                        {AddressSpaceKind::Global, AddressSpaceKind::Local})) {
    return rewriter.notifyMatchFailure(
        op,
        "only global and local memory spaces are supported by this pattern");
  }

  // Get constant offset
  int32_t off = 0;
  if (std::optional<int32_t> constOff =
          ValueOrI32::getConstant(op.getConstOffset())) {
    off = *constOff;
  } else {
    return rewriter.notifyMatchFailure(
        op, "only constant offsets are supported by this pattern");
  }

  Location loc = op.getLoc();
  TypedValue<RegisterTypeInterface> data = op.getValue();
  Value addr = op.getAddr();
  Value offset = op.getOffset();
  RegisterTypeInterface dataTy = data.getType();
  Type addrTy = addr.getType();

  // Check if the offset is constant and add it to the constant offset.
  if (std::optional<int32_t> constOff = ValueOrI32::getConstant(offset)) {
    off += *constOff;
    offset = nullptr;
  }

  // Number of 32-bit words to store
  int16_t numWords = dataTy.getAsRange().size();

  // Handle local memory store (DS)
  if (space == AddressSpaceKind::Local) {
    auto vgprAddrTy = dyn_cast<VGPRType>(addrTy);
    if (!vgprAddrTy) {
      return rewriter.notifyMatchFailure(
          op, "expected VGPR address for store to shared memory space");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op,
                                         "only constant offsets are supported "
                                         "for store to shared memory space");
    }
    offset = getI32Constant(rewriter, loc, off);

    // Convert data to VGPRType if needed
    Value dataRange = data;

    switch (numWords) {
    case 1:
      DS_WRITE_B32::create(rewriter, loc, dataRange, addr, offset);
      break;
    case 2:
      DS_WRITE_B64::create(rewriter, loc, dataRange, addr, offset);
      break;
    case 3:
      DS_WRITE_B96::create(rewriter, loc, dataRange, addr, offset);
      break;
    case 4:
      DS_WRITE_B128::create(rewriter, loc, dataRange, addr, offset);
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for store to shared memory space");
    }
    rewriter.eraseOp(op);
    return success();
  }

  // Handle global memory store
  auto addrRegTy = dyn_cast<RegisterTypeInterface>(addrTy);
  if (!addrRegTy) {
    return rewriter.notifyMatchFailure(
        op, "expected register type address for store to global memory space");
  }

  bool addrIsSGPR = isSGPR(addrRegTy, 2);
  bool addrIsVGPR = isVGPR(addrRegTy, 2);

  // Handle SMEM store (SGPR address, SGPR data)
  if (addrIsSGPR && isSGPR(dataTy, -1)) {
    if (offset && !isSGPR(offset.getType(), 1)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected SGPR offset for SMEM store");
    }
    if (offset) {
      return rewriter.notifyMatchFailure(op, "nyi: SGPR offset for SMEM store");
    }

    switch (numWords) {
    case 1:
      S_STORE_DWORD::create(rewriter, loc, data, addr, nullptr,
                            getI32Constant(rewriter, loc, off));
      break;
    case 2:
      S_STORE_DWORDX2::create(rewriter, loc, data, addr, nullptr,
                              getI32Constant(rewriter, loc, off));
      break;
    case 4:
      S_STORE_DWORDX4::create(rewriter, loc, data, addr, nullptr,
                              getI32Constant(rewriter, loc, off));
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for SMEM store");
    }
    rewriter.eraseOp(op);
    return success();
  }

  // Handle VMEM store (global_store)
  if (!addrIsVGPR && !addrIsSGPR) {
    return rewriter.notifyMatchFailure(
        op, "expected VGPR or SGPR address for store to global memory space");
  }
  if (addrIsVGPR && offset) {
    return rewriter.notifyMatchFailure(
        op, "expected no offset with VGPR address for global store");
  }
  if (addrIsSGPR && offset && !isVGPR(offset.getType(), 1)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected VGPR offset for store to "
                                       "global memory space with SGPR address");
  }
  Value cOff = nullptr;
  if (off > 0) {
    cOff = getI32Constant(rewriter, loc, off);
  }
  switch (numWords) {
  case 1:
    GLOBAL_STORE_DWORD::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 2:
    GLOBAL_STORE_DWORDX2::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 3:
    GLOBAL_STORE_DWORDX3::create(rewriter, loc, data, addr, offset, cOff);
    break;
  case 4:
    GLOBAL_STORE_DWORDX4::create(rewriter, loc, data, addr, offset, cOff);
    break;
  default:
    return rewriter.notifyMatchFailure(
        op, "unsupported number of words for store to global memory space");
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// SubIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult SubIOpPattern::matchAndRewrite(lsir::SubIOp op,
                                             PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getDst().getType();
  Value dst = op.getDst();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  OperandKind kind = getOperandKind(oTy);
  unsigned width = op.getSemantics().getWidth();
  OperandKind lhsKind, rhsKind;

  // Check we can transform this op
  if (failed(checkAIOp(op, rewriter, kind, lhs, rhs, oTy, width, lhsKind,
                       rhsKind, {16, 32, 64}, {16, 32, 64})))
    return failure();

  Location loc = op.getLoc();
  // Maybe split operands if they are register ranges
  ValueRange dstR = splitRange(rewriter, loc, dst);
  ValueRange lhsR = splitRange(rewriter, loc, lhs);
  ValueRange rhsR = splitRange(rewriter, loc, rhs);

  // Move operand to lhs if needed
  if (kind == OperandKind::VGPR &&
      isOperand(rhsKind, {OperandKind::IntImm, OperandKind::SGPR})) {
    std::swap(lhs, rhs);
    std::swap(lhsR, rhsR);
    std::swap(lhsKind, rhsKind);
  }

  // At this point, operands are valid - create the appropriate add op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result = createSOP2Out2In2<SSubU32>(rewriter, loc, dst, lhs, rhs);
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo = createSOP2Out2In2<SSubU32>(
        rewriter, loc, getElemOr(dstR, 0, dst), getElemOr(lhsR, 0, lhs),
        getElemOr(rhsR, 0, rhs));
    Value hi = createSOP2Out2In3<SSubbU32>(
        rewriter, loc, getElemOr(dstR, 1, dst), getElemOr(lhsR, 1, lhs),
        getElemOr(rhsR, 1, rhs));
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Handle the VGPR case
  if (width <= 16) {
    Value result = createNewVOP<VSubU16>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createNewVOP<VSubU32>(rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR sub
  Value carry = createAllocation(
      rewriter, loc, rewriter.getType<SGPRType>(RegisterRange(Register(), 2)));

  Value lo = VSubCoU32::create(rewriter, loc, getElemOr(dstR, 0, dst), carry,
                               getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs))
                 .getDst0Res();
  Value hi = VSubbCoU32::create(rewriter, loc, getElemOr(dstR, 1, dst), carry,
                                getElemOr(lhsR, 1, lhs),
                                getElemOr(rhsR, 1, rhs), carry)
                 .getDst0Res();
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
  return success();
}

//===----------------------------------------------------------------------===//
// WaitOpPattern
//===----------------------------------------------------------------------===//

/// Enum to classify memory operation types for wait count purposes.
enum class MemOpKind {
  Unknown,
  VMEM, // Global load/store
  SMEM, // Scalar memory load/store
  DS,   // Local data share read/write
};

/// Classify an operation to determine which wait counter it affects.
template <typename OpTy>
static MemOpKind classifyMemOp(Operation *op) {
  auto mOp = dyn_cast<OpTy>(op);
  if (!mOp)
    return MemOpKind::Unknown;
  auto memSpace = dyn_cast<amdgcn::AddressSpaceAttr>(mOp.getMemorySpace());
  if (!memSpace)
    return MemOpKind::Unknown;
  AddressSpaceKind space = memSpace.getSpace();
  if (space == AddressSpaceKind::Local)
    return MemOpKind::DS;
  if (space != AddressSpaceKind::Global)
    return MemOpKind::Unknown;
  if (isSGPR(mOp.getAddr().getType(), 2) &&
      (isSGPR(mOp.getOffset().getType(), 1) ||
       ValueOrI32::getConstant(mOp.getOffset())))
    return MemOpKind::SMEM;
  return MemOpKind::VMEM;
}

static MemOpKind classifyMemOp(Operation *op) {
  MemOpKind kind = classifyMemOp<lsir::LoadOp>(op);
  if (kind != MemOpKind::Unknown)
    return kind;
  kind = classifyMemOp<lsir::StoreOp>(op);
  if (kind != MemOpKind::Unknown)
    return kind;
  return MemOpKind::Unknown;
}

LogicalResult WaitOpPattern::matchAndRewrite(lsir::WaitOp op,
                                             PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  int32_t vmcnt = 0;
  int32_t lgkmcnt = 0;
  int32_t expcnt = 0;

  bool hasUnknown = false;

  SmallVector<Operation *> depOps;
  for (Value dep : op.getDependencies()) {
    Operation *definingOp = dep.getDefiningOp();
    if (!definingOp) {
      hasUnknown = true;
      break;
    }
    depOps.push_back(definingOp);
    MemOpKind kind = classifyMemOp(definingOp);
    switch (kind) {
    case MemOpKind::VMEM:
      ++vmcnt;
      break;
    case MemOpKind::SMEM:
    case MemOpKind::DS:
      ++lgkmcnt;
      break;
    case MemOpKind::Unknown:
      hasUnknown = true;
      break;
    }
    if (hasUnknown)
      break;
  }
  // If there are any unknown dependencies, we have to wait for all kinds.
  if (hasUnknown) {
    vmcnt = -1;
    lgkmcnt = -1;
    expcnt = -1;
  }
  auto getCnt = [&](int32_t count) -> IntegerAttr {
    if (count < 0)
      return rewriter.getI8IntegerAttr(0);
    return count == 0 ? IntegerAttr() : rewriter.getI8IntegerAttr(count);
  };
  SWaitcnt::create(rewriter, loc, getCnt(vmcnt), getCnt(expcnt),
                   getCnt(lgkmcnt));
  rewriter.eraseOp(op);
  for (Operation *depOp : depOps) {
    // Mark dependency operations as used.
    rewriter.modifyOpInPlace(depOp, []() {});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpPattern
//===----------------------------------------------------------------------===//

/// Create ptr + offset with sign-extension to i64.
static Value addPtrWithSignExtend(PatternRewriter &rewriter, Location loc,
                                  Value ptr, Value offset,
                                  RegisterTypeInterface resultTy,
                                  RegisterTypeInterface extDstTy) {
  auto i64Semantics = TypeAttr::get(rewriter.getI64Type());
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value extDst = createAllocation(rewriter, loc, extDstTy);
  Value extVal = lsir::ExtSIOp::create(rewriter, loc, i64Semantics,
                                       i32Semantics, extDst, offset)
                     .getDstRes();
  Value addDst = createAllocation(rewriter, loc, resultTy);
  return lsir::AddIOp::create(rewriter, loc, i64Semantics, addDst, ptr, extVal)
      .getDstRes();
}

/// Create ptr + offset (no extension, i32).
static Value addPtrNoExtend(PatternRewriter &rewriter, Location loc, Value ptr,
                            Value offset, RegisterTypeInterface resultTy) {
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());
  Value addDst = createAllocation(rewriter, loc, resultTy);
  return lsir::AddIOp::create(rewriter, loc, i32Semantics, addDst, ptr, offset)
      .getDstRes();
}

LogicalResult
PtrAddOpPattern::matchAndRewrite(PtrAddOp op, PatternRewriter &rewriter) const {
  TypedValue<AMDGCNRegisterTypeInterface> ptr = op.getPtr();
  TypedValue<VGPRType> dynamicOffset = op.getDynamicOffset();
  TypedValue<SGPRType> uniformOffset = op.getUniformOffset();
  int64_t constOffset = op.getConstOffset();

  // Bail if the offsets are not 32-bit.
  if (uniformOffset && uniformOffset.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "uniform offset must have register size 1");
  }
  if (dynamicOffset && dynamicOffset.getType().getAsRange().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "dynamic offset must have register size 1");
  }

  // Trivial case: no offsets.
  if (!dynamicOffset && !uniformOffset && constOffset == 0) {
    rewriter.replaceOp(op, ptr);
    return success();
  }

  // Get the types and context.
  AMDGCNRegisterTypeInterface ptrTy = ptr.getType();
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  RegisterTypeInterface resultTy = op.getResult().getType();
  bool ptrIsSGPR = ptrTy.getRegisterKind() == RegisterKind::SGPR;
  auto i32Semantics = TypeAttr::get(rewriter.getI32Type());

  // Compute uniform + const (i32) or null.
  Value uniformVal;
  if (uniformOffset && constOffset != 0) {
    Value cst =
        getI32Constant(rewriter, loc, static_cast<int32_t>(constOffset));
    Value dst = createAllocation(rewriter, loc, getSGPR(ctx, 1));
    uniformVal = lsir::AddIOp::create(rewriter, loc, i32Semantics, dst,
                                      uniformOffset, cst)
                     .getDstRes();
  } else if (uniformOffset) {
    uniformVal = uniformOffset;
  } else if (constOffset != 0) {
    uniformVal =
        getI32Constant(rewriter, loc, static_cast<int32_t>(constOffset));
  }

  // If ptr is SGPR, compute (ptr + signExtendToI64(uniform + const)) + dynamic.
  // TODO: Add flags to `ptr_add` as we then could avoid sign extension.
  if (ptrIsSGPR) {
    Value base = ptr;
    if (uniformVal) {
      // If the offset is a signless integer, we need to put it in an SGPR.
      if (uniformVal.getType().isSignlessInteger()) {
        Value alloc = createAllocation(rewriter, loc, getSGPR(ctx, 1));
        uniformVal =
            lsir::MovOp::create(rewriter, loc, alloc, uniformVal).getDstRes();
      }
      base = addPtrWithSignExtend(rewriter, loc, base, uniformVal, ptrTy,
                                  getSGPR(ctx, 2));
    }
    if (dynamicOffset) {
      base = addPtrWithSignExtend(rewriter, loc, base, dynamicOffset, resultTy,
                                  getVGPR(ctx, 2));
    }
    rewriter.replaceOp(op, base);
    return success();
  }

  // The ptr is a VGPR, compute (ptr + ((uniform + const) + dynamic)).
  Value offsetVal = uniformVal;

  if (offsetVal && dynamicOffset) {
    offsetVal =
        lsir::AddIOp::create(rewriter, loc, i32Semantics,
                             createAllocation(rewriter, loc, getVGPR(ctx, 1)),
                             offsetVal, dynamicOffset)
            .getDstRes();
  } else if (dynamicOffset) {
    offsetVal = dynamicOffset;
  }
  assert(offsetVal && "offsetVal must be non-null");

  // If the ptr is a single VGPR we do a simple addition without sign extension.
  if (ptrTy.getAsRange().size() == 1) {
    Value result = addPtrNoExtend(rewriter, loc, ptr, offsetVal, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  // If the offset is a signless integer, we need to put it in an SGPR.
  if (offsetVal.getType().isSignlessInteger()) {
    Value alloc = createAllocation(rewriter, loc, getSGPR(ctx, 1));
    offsetVal =
        lsir::MovOp::create(rewriter, loc, alloc, offsetVal).getDstRes();
  }

  // Compute the result.
  Value result = addPtrWithSignExtend(rewriter, loc, ptr, offsetVal, resultTy,
                                      getVGPR(ctx, 2));
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ToAMDGCNPass patterns
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::populateToAMDGCNPatterns(
    RewritePatternSet &patterns) {
  patterns.add< // Arithmetic ops.
      AddFOpPattern, AddIOpPattern, AndIOpPattern, ExtSIOpPattern,
      ExtUIOpPattern, MaximumFOpPattern, MinimumFOpPattern, MulFOpPattern,
      MulIOpPattern, MulHiSIOpPattern, OrIOpPattern, ShLIOpPattern,
      ShRSIOpPattern, ShRUIOpPattern, SubFOpPattern, SubIOpPattern,
      XOrIOpPattern,
      // Memory ops.
      AllocaOpPattern, AssumeNoaliasOpPattern, LoadOpPattern, StoreOpPattern,
      // Data movement ops.
      MovOpPattern, RegCastOpPattern,
      // Pointer ops.
      PtrAddOpPattern,
      // Control ops.
      KernelOpPattern, ReturnOpPattern,
      // Synchronization ops.
      WaitOpPattern>(patterns.getContext());
}
