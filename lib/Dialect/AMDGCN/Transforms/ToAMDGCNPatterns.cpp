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
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/ValueOrConst.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <utility>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

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

struct MulIOpPattern : public OpRewritePattern<lsir::MulIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(lsir::MulIOp op,
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

// Create VOp2/VOp3 operation.
template <OpCode vop2C, OpCode vop3C>
static Value createVOP(bool isVOp3, OpBuilder &builder, Location loc, Value dst,
                       Value src1, Value src2, Value vdst1 = nullptr,
                       Value src3 = nullptr) {
  return !isVOp3 ? inst::VOP2Op::create(builder, loc, vop2C, dst, vdst1, src1,
                                        src2, src3)
                       .getVdst0Res()
                 : inst::VOP3Op::create(builder, loc, vop3C, dst, vdst1, src1,
                                        src2, src3)
                       .getVdst0Res();
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
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_ADD_U32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo =
        S_ADD_U32::create(rewriter, loc, getElemOr(dstR, 0, dst),
                          getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs))
            .getSdstRes();
    Value hi =
        S_ADDC_U32::create(rewriter, loc, getElemOr(dstR, 1, dst),
                           getElemOr(lhsR, 1, lhs), getElemOr(rhsR, 1, rhs))
            .getSdstRes();
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Handle the VGPR case
  if (width <= 16) {
    Value result = createVOP<OpCode::V_ADD_U16, OpCode::V_ADD_U16_E64>(
        isVOp3, rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createVOP<OpCode::V_ADD_U32, OpCode::V_ADD_U32_E64>(
        isVOp3, rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR add
  Value result = V_LSHL_ADD_U64::create(rewriter, loc, dst, lhs,
                                        getI32Constant(rewriter, loc, 0), rhs)
                     .getVdst0Res();
  rewriter.replaceOp(op, result);
  return success();
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
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_AND_B32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = S_AND_B64::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
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
  Value result = createVOP<OpCode::V_AND_B32, OpCode::V_AND_B32_E64>(
      isVOp3, rewriter, loc, dst, lhs, rhs);
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
  if (!llvm::all_of(gpuAbiTy.getInputs(),
                    llvm::IsaPred<SGPRType, SGPRRangeType>))
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
  for (auto [i, arg] : llvm::reverse(llvm::enumerate(entry->getArguments()))) {
    if (arg.use_empty())
      continue;
    Value rA = LoadArgOp::create(rewriter, op.getLoc(), arg.getType(), i);
    rewriter.replaceAllUsesWith(arg, rA);
  }
  entry->eraseArguments(0, op.getNumArguments());

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
          DS_READ_B32::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 2:
      result =
          DS_READ_B64::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 3:
      result =
          DS_READ_B96::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    case 4:
      result =
          DS_READ_B128::create(rewriter, loc, dst, addr, offset).getResult();
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported number of words for load from shared memory space");
    }
    rewriter.replaceAllUsesWith(op.getDstRes(), result);
    rewriter.eraseOp(op);
    return success();
  }

  // Handle a SMEM load
  bool addrIsSGPR = isSGPR(addrTy, 2);
  if (addrIsSGPR && (!offset || isSGPR(offset.getType(), 1))) {
    if (offset) {
      return rewriter.notifyMatchFailure(op, "nyi: SGPR offset");
    }
    switch (numWords) {
    case 1:
      result = S_LOAD_DWORD::create(rewriter, loc, dst, addr, off).getResult();
      break;
    case 2:
      result =
          S_LOAD_DWORDX2::create(rewriter, loc, dst, addr, off).getResult();
      break;
    case 4:
      result =
          S_LOAD_DWORDX4::create(rewriter, loc, dst, addr, off).getResult();
      break;
    case 8:
      result =
          S_LOAD_DWORDX8::create(rewriter, loc, dst, addr, off).getResult();
      break;
    case 16:
      result =
          S_LOAD_DWORDX16::create(rewriter, loc, dst, addr, off).getResult();
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
      result = GLOBAL_LOAD_DWORD::create(rewriter, loc, dst, addr, offset, off)
                   .getResult();
      break;
    case 2:
      result =
          GLOBAL_LOAD_DWORDX2::create(rewriter, loc, dst, addr, offset, off)
              .getResult();
      break;
    case 3:
      result =
          GLOBAL_LOAD_DWORDX3::create(rewriter, loc, dst, addr, offset, off)
              .getResult();
      break;
    case 4:
      result =
          GLOBAL_LOAD_DWORDX4::create(rewriter, loc, dst, addr, offset, off)
              .getResult();
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
  RegisterTypeInterface oTy = op.getType();
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
      lhs = V_MOV_B32_E32::create(rewriter, loc, vgpr, lhs);
      lhsKind = OperandKind::VGPR;
    }
  }

  // At this point, operands are valid - create the appropriate mul op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result =
          S_MUL_I32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
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

    dHi = S_MUL_I32::create(rewriter, loc, dHi, rLo, lHi).getSdstRes();
    t0 = S_MUL_HI_U32::create(rewriter, loc, t0, rLo, lLo).getSdstRes();
    t1 = S_MUL_I32::create(rewriter, loc, t1, rHi, lLo).getSdstRes();
    dHi = S_ADD_I32::create(rewriter, loc, dHi, t0, dHi).getSdstRes();
    dLo = S_MUL_I32::create(rewriter, loc, dLo, rLo, lLo).getSdstRes();
    dHi = S_ADD_I32::create(rewriter, loc, dHi, dHi, t1).getSdstRes();

    // Combine low and high parts
    Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dLo, dHi});
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width <= 16) {
    Value result = createVOP<OpCode::V_MUL_LO_U16, OpCode::V_MUL_LO_U16_E64>(
        isVOp3, rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    if (lhsKind == OperandKind::IntImm) {
      APInt constVal;
      if (matchPattern(lhs, m_ConstantInt(&constVal))) {
        Value vgpr = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
        lhs = V_MOV_B32_E32::create(rewriter, loc, vgpr, lhs);
        lhsKind = OperandKind::VGPR;
      }
    }
    Value result =
        V_MUL_LO_U32::create(rewriter, loc, dst, lhs, rhs).getVdst0Res();
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR multiplication
  Value lLo = getElemOr(lhsR, 0, lhs);
  Value lHi = getElemOr(lhsR, 1, lhs);
  Value rLo = getElemOr(rhsR, 0, rhs);
  Value rHi = getElemOr(rhsR, 1, rhs);

  // Allocate temporaries
  Value t0 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
  Value t1 = createAllocation(rewriter, loc, getVGPR(getCtx(rewriter)));
  Value carry = createAllocation(rewriter, loc, getSGPR(getCtx(rewriter), 2));
  t0 = V_MUL_LO_U32::create(rewriter, loc, t0, rHi, lLo).getVdst0Res();
  t1 = V_MUL_LO_U32::create(rewriter, loc, t1, rLo, lHi).getVdst0Res();
  Value zero = getI32Constant(rewriter, loc, 0);
  ValueRange dT0 = splitRange(
      rewriter, loc,
      V_MAD_U64_U32::create(rewriter, loc, dst, carry, rLo, lLo, zero)
          .getVdst0Res());
  Value t3 =
      V_ADD3_U32::create(rewriter, loc, dT0[1], dT0[1], t1, t0).getVdst0Res();
  Value result = MakeRegisterRangeOp::create(rewriter, loc, oTy, {dT0[0], t3});
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// OrIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult OrIOpPattern::matchAndRewrite(lsir::OrIOp op,
                                            PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_OR_B32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result = S_OR_B64::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  Value result = createVOP<OpCode::V_OR_B32, OpCode::V_OR_B32_E64>(
      isVOp3, rewriter, loc, dst, lhs, rhs);
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

  OperandKind kind = getOperandKind(op.getType());
  if (kind != OperandKind::VGPR)
    return rewriter.notifyMatchFailure(op, "only VGPR mov is supported");

  Value res =
      V_MOV_B32_E32::create(rewriter, op.getLoc(), op.getDst(), op.getValue());
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

  Value res = V_MOV_B32_E32::create(
      rewriter, loc, createAllocation(rewriter, loc, op.getType()),
      op.getSrc());
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
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_LSHL_B32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result =
        S_LSHL_B64::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createVOP<OpCode::V_LSHLREV_B16, OpCode::V_LSHLREV_B16_E64>(
        isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result =
        createVOP<OpCode::V_LSHLREV_B32_E32, OpCode::V_LSHLREV_B32_E64>(
            isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result =
      V_LSHLREV_B64::create(rewriter, loc, dst, rhs, lhs).getVdst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRSIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRSIOpPattern::matchAndRewrite(lsir::ShRSIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_ASHR_I32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result =
        S_ASHR_I64::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createVOP<OpCode::V_ASHRREV_I16, OpCode::V_ASHRREV_I16_E64>(
        isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createVOP<OpCode::V_ASHRREV_I32, OpCode::V_ASHRREV_I32_E64>(
        isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result =
      V_ASHRREV_I64::create(rewriter, loc, dst, rhs, lhs).getVdst0Res();
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ShRUIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ShRUIOpPattern::matchAndRewrite(lsir::ShRUIOp op,
                                              PatternRewriter &rewriter) const {
  RegisterTypeInterface oTy = op.getType();
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
      Value result =
          S_LSHR_B32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value result =
        S_LSHR_B64::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
    rewriter.replaceOp(op, result);
    return success();
  }

  // Handle the VGPR case
  if (width == 16) {
    // NOTE: Operands are reversed
    Value result = createVOP<OpCode::V_LSHRREV_B16, OpCode::V_LSHRREV_B16_E64>(
        isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width == 32) {
    // NOTE: Operands are reversed
    Value result = createVOP<OpCode::V_LSHRREV_B32, OpCode::V_LSHRREV_B32_E64>(
        isVOp3, rewriter, loc, dst, rhs, lhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  // NOTE: Operands are reversed
  Value result =
      V_LSHRREV_B64::create(rewriter, loc, dst, rhs, lhs).getVdst0Res();
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

    // Convert data to VGPRRangeType if needed
    Value dataRange = data;
    if (isa<VGPRType>(dataTy)) {
      dataRange = MakeRegisterRangeOp::create(rewriter, loc, dataTy, {data});
    }

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
      S_STORE_DWORD::create(rewriter, loc, data, addr, off);
      break;
    case 2:
      S_STORE_DWORDX2::create(rewriter, loc, data, addr, off);
      break;
    case 4:
      S_STORE_DWORDX4::create(rewriter, loc, data, addr, off);
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

  switch (numWords) {
  case 1:
    GLOBAL_STORE_DWORD::create(rewriter, loc, data, addr, offset, off);
    break;
  case 2:
    GLOBAL_STORE_DWORDX2::create(rewriter, loc, data, addr, offset, off);
    break;
  case 3:
    GLOBAL_STORE_DWORDX3::create(rewriter, loc, data, addr, offset, off);
    break;
  case 4:
    GLOBAL_STORE_DWORDX4::create(rewriter, loc, data, addr, offset, off);
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
  RegisterTypeInterface oTy = op.getType();
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

  // Determine whether we need to use VOP3.
  bool isVOp3 = rhsKind == OperandKind::SGPR;

  // At this point, operands are valid - create the appropriate add op
  if (kind == OperandKind::SGPR) {
    if (width == 32) {
      Value result =
          S_SUB_U32::create(rewriter, loc, dst, lhs, rhs).getSdstRes();
      rewriter.replaceOp(op, result);
      return success();
    }
    Value lo =
        S_SUB_U32::create(rewriter, loc, getElemOr(dstR, 0, dst),
                          getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs))
            .getSdstRes();
    Value hi =
        S_SUBB_U32::create(rewriter, loc, getElemOr(dstR, 1, dst),
                           getElemOr(lhsR, 1, lhs), getElemOr(rhsR, 1, rhs))
            .getSdstRes();
    rewriter.replaceOp(
        op, MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
    return success();
  }

  // Handle the VGPR case
  if (width <= 16) {
    Value result = createVOP<OpCode::V_SUB_U16, OpCode::V_SUB_U16_E64>(
        isVOp3, rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
  if (width <= 32) {
    Value result = createVOP<OpCode::V_SUB_U32, OpCode::V_SUB_U32_E64>(
        isVOp3, rewriter, loc, dst, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }

  // 64-bit VGPR sub
  Value carry = createAllocation(
      rewriter, loc,
      rewriter.getType<SGPRRangeType>(RegisterRange(Register(), 2)));

  Value lo =
      V_SUB_CO_U32_E64::create(rewriter, loc, getElemOr(dstR, 0, dst), carry,
                               getElemOr(lhsR, 0, lhs), getElemOr(rhsR, 0, rhs))
          .getVdst0Res();
  Value hi = V_SUBB_CO_U32_E64::create(rewriter, loc, getElemOr(dstR, 1, dst),
                                       carry, getElemOr(lhsR, 1, lhs),
                                       getElemOr(rhsR, 1, rhs), carry)
                 .getVdst0Res();
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, loc, oTy, {lo, hi}));
  return success();
}

//===----------------------------------------------------------------------===//
// TimingStartOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
TimingStartOpPattern::matchAndRewrite(lsir::TimingStartOp op,
                                      PatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Allocate SGPRs for timestamp (2 SGPRs for 64-bit time value)
  Value timestampSgpr = createAllocation(
      rewriter, loc,
      rewriter.getType<SGPRRangeType>(RegisterRange(Register(), 2)));

  // Read timestamp using s_memtime
  Value timestampRead = rewriter.create<inst::SMEMLoadOp>(
      loc, timestampSgpr.getType(), OpCode::S_MEMTIME, timestampSgpr, Value(),
      0);

  // Wait for memtime to complete
  S_WAITCNT::create(rewriter, loc, 0, 0, 0);

  // Replace the operation with the timestamp read result
  rewriter.replaceOp(op, timestampRead);
  return success();
}

//===----------------------------------------------------------------------===//
// TimingStopOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
TimingStopOpPattern::matchAndRewrite(lsir::TimingStopOp op,
                                     PatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Get operands
  Value startTimeSgpr = op.getStartTime();
  Value startBufferPtrSgpr = op.getStartBufferPtr();
  Value endBufferPtrSgpr = op.getEndBufferPtr();

  // Read end timestamp using s_memtime
  Value endTimestampSgpr = createAllocation(
      rewriter, loc,
      rewriter.getType<SGPRRangeType>(RegisterRange(Register(), 2)));
  Value endTimeRead = rewriter.create<inst::SMEMLoadOp>(
      loc, endTimestampSgpr.getType(), OpCode::S_MEMTIME, endTimestampSgpr,
      Value(), 0);

  // Wait for memtime to complete
  S_WAITCNT::create(rewriter, loc, 0, 0, 0);

  // Helper function to convert SGPR timestamp to VGPR and store to global
  // memory These stores happen AFTER the [start, stop) interval, so they're not
  // counted
  auto storeTimestamp = [&](Value timestampSgpr, Value bufferPtrSgpr) {
    // Convert timestamp from SGPR to VGPR for global_store
    ValueRange timestampSplit =
        SplitRegisterRangeOp::create(rewriter, loc, timestampSgpr).getResults();
    Value timestampVgprLo =
        createAllocation(rewriter, loc, rewriter.getType<VGPRType>(Register()));
    Value timestampVgprHi =
        createAllocation(rewriter, loc, rewriter.getType<VGPRType>(Register()));
    Value timestampVgprLoMoved = V_MOV_B32_E32::create(
        rewriter, loc, timestampVgprLo, timestampSplit[0]);
    Value timestampVgprHiMoved = V_MOV_B32_E32::create(
        rewriter, loc, timestampVgprHi, timestampSplit[1]);
    Value timestampVgpr = MakeRegisterRangeOp::create(
        rewriter, loc, {timestampVgprLoMoved, timestampVgprHiMoved});

    // Allocate zero offset VGPR for global_store (no per-thread offset)
    Value zeroVgpr =
        createAllocation(rewriter, loc, rewriter.getType<VGPRType>(Register()));
    Value zeroConst = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value zeroVgprMoved =
        V_MOV_B32_E32::create(rewriter, loc, zeroVgpr, zeroConst);

    // Store timestamp to global memory using SGPR address and VGPR data
    rewriter.create<inst::GlobalStoreOp>(loc, OpCode::GLOBAL_STORE_DWORDX2,
                                         timestampVgpr, bufferPtrSgpr,
                                         zeroVgprMoved, 0);
  };

  // Store both timestamps (these are outside the [start, stop) interval)
  storeTimestamp(startTimeSgpr, startBufferPtrSgpr);
  storeTimestamp(endTimeRead, endBufferPtrSgpr);

  // Wait for stores to complete
  S_WAITCNT::create(rewriter, loc, 0, 0, 0);

  // Erase the operation (it has no results)
  rewriter.eraseOp(op);
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
  inst::SWaitcntOp::create(rewriter, loc,
                           InstAttr::get(getCtx(rewriter), OpCode::S_WAITCNT),
                           getCnt(vmcnt), getCnt(expcnt), getCnt(lgkmcnt));
  rewriter.eraseOp(op);
  for (Operation *depOp : depOps) {
    // Mark dependency operations as used.
    rewriter.modifyOpInPlace(depOp, []() {});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ToAMDGCNPass patterns
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::populateToAMDGCNPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AddIOpPattern, AllocaOpPattern, AssumeNoaliasOpPattern,
               AndIOpPattern, KernelOpPattern, LoadOpPattern, MovOpPattern,
               MulIOpPattern, OrIOpPattern, RegCastOpPattern, ReturnOpPattern,
               ShLIOpPattern, ShRSIOpPattern, ShRUIOpPattern, StoreOpPattern,
               SubIOpPattern, TimingStartOpPattern, TimingStopOpPattern,
               WaitOpPattern>(patterns.getContext());
}
