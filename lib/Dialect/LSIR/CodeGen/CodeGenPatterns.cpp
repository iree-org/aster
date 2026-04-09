//===- CodeGenPatterns.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSIR CodeGen patterns
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/LSIR/CodeGen/CodeGen.h"

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::lsir;

namespace {
//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithBinaryOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// MulExtendedOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename HiOpTy>
struct MulExtendedOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy, typename NewOpTy>
struct ArithCastOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCmpIOpPattern
//===----------------------------------------------------------------------===//
struct ArithCmpIOpPattern : public OpCodeGenPattern<arith::CmpIOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(arith::CmpIOp op, arith::CmpIOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithCmpFOpPattern
//===----------------------------------------------------------------------===//
struct ArithCmpFOpPattern : public OpCodeGenPattern<arith::CmpFOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(arith::CmpFOp op, arith::CmpFOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// CFCondBranchOpPattern
//===----------------------------------------------------------------------===//
struct CFCondBranchOpPattern : public OpCodeGenPattern<cf::CondBranchOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// CFBranchOpPattern
//===----------------------------------------------------------------------===//
struct CFBranchOpPattern : public OpCodeGenPattern<cf::BranchOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// KernelOpConversion
//===----------------------------------------------------------------------===//
/// Converts block argument types in amdgcn.kernel regions.
/// Similar to FuncOpConversion but for kernel ops which have NoRegionArguments.
struct KernelOpConversion : public OpCodeGenPattern<amdgcn::KernelOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(amdgcn::KernelOp op, amdgcn::KernelOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ArithMinMaxOpPattern -- lower arith.min/maxui/si to cmpi + select
//===----------------------------------------------------------------------===//
template <typename MinMaxOp, arith::CmpIPredicate pred>
struct ArithMinMaxOpPattern : public OpCodeGenPattern<MinMaxOp> {
  using OpCodeGenPattern<MinMaxOp>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(MinMaxOp op, typename MinMaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    // Lower to lsir.cmpi + lsir.select directly (skip arith intermediates).
    // lsir.cmpi uses DPS: determine the compare dst type (SCC for scalar ops,
    // VCC for vector ops) from the lhs register kind.
    Type regType = this->converter.convertType(op);
    Value dst = this->createAlloca(rewriter, loc, regType);
    Type cmpDstType =
        isa<amdgcn::VGPRType>(lhs.getType())
            ? Type(amdgcn::VCCType::get(rewriter.getContext(), Register()))
            : Type(amdgcn::SCCType::get(rewriter.getContext(), Register()));
    Value cmpDst = this->createAlloca(rewriter, loc, cmpDstType);
    Value cmp = lsir::CmpIOp::create(
                    rewriter, loc, TypeAttr::get(op.getLhs().getType()),
                    arith::CmpIPredicateAttr::get(rewriter.getContext(), pred),
                    cmpDst, lhs, rhs)
                    .getDstRes();
    rewriter.replaceOpWithNewOp<lsir::SelectOp>(op, dst, cmp, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//
struct ArithSelectOpPattern : public OpCodeGenPattern<arith::SelectOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//
template <typename OpTy>
struct FromToRegOpPattern : public OpCodeGenPattern<OpTy> {
  using OpCodeGenPattern<OpTy>::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// RegConstraintPattern
//===----------------------------------------------------------------------===//
struct RegConstraintPattern : public OpCodeGenPattern<RegConstraintOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AssumeRangeOpPattern
//===----------------------------------------------------------------------===//
struct AssumeRangeOpPattern
    : public OpCodeGenPattern<aster_utils::AssumeRangeOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AssumeUniformOpPattern
//===----------------------------------------------------------------------===//
struct AssumeUniformOpPattern
    : public OpCodeGenPattern<aster_utils::AssumeUniformOp> {
  using OpCodeGenPattern::OpCodeGenPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ArithBinaryOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult ArithBinaryOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  rewriter.replaceOpWithNewOp<NewOpTy>(op, TypeAttr::get(op.getType()), dst,
                                       adaptor.getLhs(), adaptor.getRhs());
  return success();
}

//===----------------------------------------------------------------------===//
// MulExtendedOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename HiOpTy>
LogicalResult MulExtendedOpPattern<OpTy, HiOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Only handle the case where the low result is unused (the high result is
  // what division-by-constant needs). If the low result has users, bail --
  // arith canonicalization would have folded it to arith.muli.
  if (!op.getLow().use_empty())
    return rewriter.notifyMatchFailure(op, "low result has users");

  Location loc = op.getLoc();
  Type type = this->converter.convertType(op.getHigh().getType());
  auto semantics = TypeAttr::get(op.getHigh().getType());
  Value dst = this->createAlloca(rewriter, loc, type);
  Value hi = HiOpTy::create(rewriter, loc, semantics, dst, adaptor.getLhs(),
                            adaptor.getRhs())
                 .getDstRes();
  rewriter.replaceOp(op, {nullptr, hi});
  return success();
}

//===----------------------------------------------------------------------===//
// ArithCastOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy, typename NewOpTy>
LogicalResult ArithCastOpPattern<OpTy, NewOpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);

  // Get the element type from the original operation
  Type srcElemType = getElementTypeOrSelf(op.getIn().getType());
  Type dstElemType = getElementTypeOrSelf(op.getType());

  rewriter.replaceOpWithNewOp<NewOpTy>(op, TypeAttr::get(dstElemType),
                                       TypeAttr::get(srcElemType), dst,
                                       adaptor.getIn());
  return success();
}

//===----------------------------------------------------------------------===//
// ArithCmpIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ArithCmpIOpPattern::matchAndRewrite(arith::CmpIOp op,
                                    arith::CmpIOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Type dstType = converter.convertType(op.getResult());
  Value dst = createAlloca(rewriter, op.getLoc(), dstType);
  auto cmpOp = lsir::CmpIOp::create(
      rewriter, op.getLoc(), TypeAttr::get(op.getLhs().getType()),
      op.getPredicateAttr(), dst, adaptor.getLhs(), adaptor.getRhs());
  rewriter.replaceOp(op, cmpOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ArithCmpFOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ArithCmpFOpPattern::matchAndRewrite(arith::CmpFOp op,
                                    arith::CmpFOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Type dstType = amdgcn::VCCType::get(op.getContext(), Register());
  Value dst = createAlloca(rewriter, op.getLoc(), dstType);
  auto cmpOp = lsir::CmpFOp::create(
      rewriter, op.getLoc(), TypeAttr::get(op.getLhs().getType()),
      op.getPredicateAttr(), dst, adaptor.getLhs(), adaptor.getRhs());
  rewriter.replaceOp(op, cmpOp);
  return success();
}

/// Convert operands to match the expected (converted) block argument types.
/// For scalar types that should become registers, insert alloca+mov.
static SmallVector<Value>
convertBranchOperands(ValueRange operands, Block *destBlock,
                      const TypeConverter &converter,
                      ConversionPatternRewriter &rewriter, Location loc) {
  SmallVector<Value> converted;
  for (auto [operand, blockArg] :
       llvm::zip(operands, destBlock->getArguments())) {
    // Get the expected converted type for this block argument
    Type expectedType = converter.convertType(blockArg.getType());
    if (!expectedType)
      expectedType = blockArg.getType();

    if (operand.getType() == expectedType) {
      converted.push_back(operand);
    } else if (isa<RegisterTypeInterface>(expectedType) &&
               operand.getType().isIntOrIndexOrFloat()) {
      // Scalar to register conversion
      Value dst = lsir::AllocaOp::create(rewriter, loc, expectedType);
      Value reg = lsir::MovOp::create(rewriter, loc, dst, operand).getDstRes();
      converted.push_back(reg);
    } else if (isa<RegisterTypeInterface>(expectedType) &&
               isa<RegisterTypeInterface>(operand.getType())) {
      // Register to register cast
      Value reg = lsir::RegCastOp::create(rewriter, loc, expectedType, operand)
                      .getResult();
      converted.push_back(reg);
    } else {
      converted.push_back(operand);
    }
  }
  return converted;
}

//===----------------------------------------------------------------------===//
// CFCondBranchOpPattern
//===----------------------------------------------------------------------===//

LogicalResult CFCondBranchOpPattern::matchAndRewrite(
    cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // The condition must already be a register type (from lsir.cmpi). If it is
  // still i1 (e.g. from lsir.cmpf), we cannot convert this op.
  Value cond = adaptor.getCondition();
  if (!isa<RegisterTypeInterface>(cond.getType()))
    return failure();

  // Convert operands to match the expected converted block argument types.
  SmallVector<Value> trueOperands =
      convertBranchOperands(adaptor.getTrueDestOperands(), op.getTrueDest(),
                            *getTypeConverter(), rewriter, loc);
  SmallVector<Value> falseOperands =
      convertBranchOperands(adaptor.getFalseDestOperands(), op.getFalseDest(),
                            *getTypeConverter(), rewriter, loc);

  rewriter.replaceOpWithNewOp<lsir::CondBranchOp>(
      op, cond, op.getTrueDest(), trueOperands, op.getFalseDest(),
      falseOperands);
  return success();
}

//===----------------------------------------------------------------------===//
// CFBranchOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
CFBranchOpPattern::matchAndRewrite(cf::BranchOp op,
                                   cf::BranchOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Convert operands to match the expected converted block argument types.
  SmallVector<Value> destOperands =
      convertBranchOperands(adaptor.getDestOperands(), op.getDest(),
                            *getTypeConverter(), rewriter, loc);

  rewriter.replaceOpWithNewOp<lsir::BranchOp>(op, op.getDest(), destOperands);
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
KernelOpConversion::matchAndRewrite(amdgcn::KernelOp op,
                                    amdgcn::KernelOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // KernelOp has NoRegionArguments, so no function signature to convert.
  // But we need to convert all block argument types in the body region
  // (e.g., loop header blocks created by SCF-to-CF conversion).
  if (failed(rewriter.convertRegionTypes(&op.getBodyRegion(),
                                         *getTypeConverter())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ArithSelectOpPattern
//===----------------------------------------------------------------------===//

LogicalResult ArithSelectOpPattern::matchAndRewrite(
    arith::SelectOp op, arith::SelectOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // lsir.select requires a register condition. If the condition is still i1
  // (e.g. from lsir.cmpf), we cannot convert this op.
  Value cond = adaptor.getCondition();
  if (!isa<RegisterTypeInterface>(cond.getType()))
    return failure();
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  rewriter.replaceOpWithNewOp<lsir::SelectOp>(
      op, dst, cond, adaptor.getTrueValue(), adaptor.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// FromToRegOpPattern
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult FromToRegOpPattern<OpTy>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  // If the input is a constant, create a mov to the proper register type.
  // Note: getDefiningOp() returns nullptr for block arguments.
  if (Operation *defOp = input.getDefiningOp();
      defOp && m_Constant().match(defOp)) {
    Type type = this->converter.convertType(op);
    Value dst = this->createAlloca(rewriter, op.getLoc(), type);
    rewriter.replaceOpWithNewOp<lsir::MovOp>(op, dst, input);
    return success();
  }
  rewriter.replaceOp(op, input);
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::aster::lsir::getDependentCodeGenDialects(DialectRegistry &registry) {
  registry.insert<lsir::LSIRDialect>();
}

void mlir::aster::lsir::populateCodeGenPatterns(CodeGenConverter &converter,
                                                RewritePatternSet &patterns,
                                                ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<lsir::LSIRDialect>();
  target.addIllegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });
  target.addDynamicallyLegalOp<RegConstraintOp>(
      [&](RegConstraintOp op) { return converter.isLegal(op); });
  target.addIllegalOp<aster_utils::AssumeRangeOp, aster_utils::AssumeUniformOp,
                      lsir::FromRegOp, lsir::ToRegOp, lsir::RegConstraintOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // arith.cmpi is converted to lsir.cmpi (DPS, returns SCC/VCC register).
  // arith.cmpf is converted to lsir.cmpf (returns i1 for now).
  target.addIllegalOp<arith::CmpIOp, arith::CmpFOp>();

  // CF dialect branch ops are always illegal — they must be replaced by the
  // corresponding lsir.br / lsir.cond_br ops that carry register conditions.
  target.addIllegalOp<cf::CondBranchOp, cf::BranchOp>();

  // KernelOp is dynamically legal - it becomes legal once the
  // KernelOpConversion pattern has converted all block argument types.
  // Start as illegal to ensure the pattern runs.
  target.addDynamicallyLegalOp<amdgcn::KernelOp>([&](amdgcn::KernelOp op) {
    // Check if any block in the body has non-register arguments. Token types
    // are always legal.
    for (Block &block : op.getBodyRegion()) {
      for (BlockArgument arg : block.getArguments()) {
        Type t = arg.getType();
        if (!isa<RegisterTypeInterface, TokenDependencyTypeInterface>(t))
          return false;
      }
    }
    return true;
  });

  // Add the patterns.
  patterns.add<ArithBinaryOpPattern<arith::AddIOp, lsir::AddIOp>,
               ArithBinaryOpPattern<arith::SubIOp, lsir::SubIOp>,
               ArithBinaryOpPattern<arith::MulIOp, lsir::MulIOp>,
               MulExtendedOpPattern<arith::MulSIExtendedOp, lsir::MulHiSIOp>,
               ArithBinaryOpPattern<arith::DivSIOp, lsir::DivSIOp>,
               ArithBinaryOpPattern<arith::DivUIOp, lsir::DivUIOp>,
               ArithBinaryOpPattern<arith::RemSIOp, lsir::RemSIOp>,
               ArithBinaryOpPattern<arith::RemUIOp, lsir::RemUIOp>,
               ArithBinaryOpPattern<arith::AndIOp, lsir::AndIOp>,
               ArithBinaryOpPattern<arith::OrIOp, lsir::OrIOp>,
               ArithBinaryOpPattern<arith::XOrIOp, lsir::XOrIOp>,
               ArithBinaryOpPattern<arith::ShLIOp, lsir::ShLIOp>,
               ArithBinaryOpPattern<arith::ShRSIOp, lsir::ShRSIOp>,
               ArithBinaryOpPattern<arith::ShRUIOp, lsir::ShRUIOp>,
               ArithBinaryOpPattern<arith::MaxSIOp, lsir::MaxSIOp>,
               ArithBinaryOpPattern<arith::MaxUIOp, lsir::MaxUIOp>,
               ArithBinaryOpPattern<arith::AddFOp, lsir::AddFOp>,
               ArithBinaryOpPattern<arith::SubFOp, lsir::SubFOp>,
               ArithBinaryOpPattern<arith::MulFOp, lsir::MulFOp>,
               ArithBinaryOpPattern<arith::DivFOp, lsir::DivFOp>,
               ArithBinaryOpPattern<arith::MaximumFOp, lsir::MaximumFOp>,
               ArithBinaryOpPattern<arith::MinimumFOp, lsir::MinimumFOp>,
               ArithCastOpPattern<arith::ExtSIOp, lsir::ExtSIOp>,
               ArithCastOpPattern<arith::ExtUIOp, lsir::ExtUIOp>,
               ArithCastOpPattern<arith::TruncIOp, lsir::TruncIOp>,
               ArithCastOpPattern<arith::ExtFOp, lsir::ExtFOp>,
               ArithCastOpPattern<arith::TruncFOp, lsir::TruncFOp>,
               ArithCastOpPattern<arith::FPToSIOp, lsir::FPToSIOp>,
               ArithCastOpPattern<arith::FPToUIOp, lsir::FPToUIOp>,
               ArithCastOpPattern<arith::SIToFPOp, lsir::SIToFPOp>,
               ArithCastOpPattern<arith::UIToFPOp, lsir::UIToFPOp>,
               ArithMinMaxOpPattern<arith::MinUIOp, arith::CmpIPredicate::ult>,
               ArithMinMaxOpPattern<arith::MinSIOp, arith::CmpIPredicate::slt>,
               ArithMinMaxOpPattern<arith::MaxUIOp, arith::CmpIPredicate::ugt>,
               ArithMinMaxOpPattern<arith::MaxSIOp, arith::CmpIPredicate::sgt>,
               ArithSelectOpPattern,
               // ASTER-specific abstractions used to connect pieces in
               // composable fashion.
               FromToRegOpPattern<ToRegOp>, FromToRegOpPattern<FromRegOp>,
               RegConstraintPattern, AssumeRangeOpPattern,
               // These patterns go together for proper composable control-flow
               // support. CF patterns need the type converter to handle block
               // argument conversion. KernelOp conversion handles block
               // argument types in kernel bodies. arith.cmpi is converted to
               // lsir.cmpi (DPS, SCC/VCC dst); cf.br/cf.cond_br are replaced
               // by lsir.br/lsir.cond_br that carry register conditions.
               ArithCmpIOpPattern, ArithCmpFOpPattern, CFCondBranchOpPattern,
               CFBranchOpPattern, KernelOpConversion, AssumeUniformOpPattern
               // That's all folks!
               >(converter);
  // Special generic pattern: converts operations by converting
  // their result types and recreating them.
  patterns.add<GenericOpConversion<RegConstraintOp>>(converter,
                                                     patterns.getContext());
}
