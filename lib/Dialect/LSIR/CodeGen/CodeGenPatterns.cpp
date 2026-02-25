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
  // lsir.cmpi returns i1 directly, operands are converted to registers
  rewriter.replaceOpWithNewOp<lsir::CmpIOp>(
      op, rewriter.getI1Type(), TypeAttr::get(op.getLhs().getType()),
      op.getPredicateAttr(), adaptor.getLhs(), adaptor.getRhs());
  return success();
}

//===----------------------------------------------------------------------===//
// ArithCmpFOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
ArithCmpFOpPattern::matchAndRewrite(arith::CmpFOp op,
                                    arith::CmpFOp::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // lsir.cmpf returns i1 directly, operands are converted to registers
  rewriter.replaceOpWithNewOp<lsir::CmpFOp>(
      op, rewriter.getI1Type(), TypeAttr::get(op.getLhs().getType()),
      op.getPredicateAttr(), adaptor.getLhs(), adaptor.getRhs());
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

  // Convert operands to match the expected converted block argument types
  SmallVector<Value> trueOperands =
      convertBranchOperands(adaptor.getTrueDestOperands(), op.getTrueDest(),
                            *getTypeConverter(), rewriter, loc);
  SmallVector<Value> falseOperands =
      convertBranchOperands(adaptor.getFalseDestOperands(), op.getFalseDest(),
                            *getTypeConverter(), rewriter, loc);

  rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
      op, op.getCondition(), op.getTrueDest(), trueOperands, op.getFalseDest(),
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

  // Convert operands to match the expected converted block argument types
  SmallVector<Value> destOperands =
      convertBranchOperands(adaptor.getDestOperands(), op.getDest(),
                            *getTypeConverter(), rewriter, loc);

  rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(), destOperands);
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
  Type type = this->converter.convertType(op);
  Value dst = this->createAlloca(rewriter, op.getLoc(), type);
  // If the original condition comes from lsir.cmpi/cmpf (i1 result), use the
  // original value to avoid the type converter wrapping it in a cast.
  // For block-argument i1 conditions, the type converter correctly maps them
  // to register types, so we use the adapted value.
  Value cond = op.getCondition().getDefiningOp<lsir::CmpIOp>() ||
                       op.getCondition().getDefiningOp<lsir::CmpFOp>()
                   ? op.getCondition()
                   : adaptor.getCondition();
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

  // arith.cmpi/cmpf are always converted to lsir counterparts.
  // They return i1 but their operands are converted to register types.
  target.addIllegalOp<arith::CmpIOp, arith::CmpFOp>();

  // Helper to check if operands are legal for CF ops. Operands are legal if
  // they are register types OR if they come from constants (which stay scalar).
  auto cfOperandsLegal = [&](ValueRange operands) {
    return llvm::all_of(operands, [&](Value v) {
      Type t = v.getType();
      // Register types are legal
      if (isa<RegisterTypeInterface, TokenDependencyTypeInterface>(t))
        return true;
      return false;
    });
  };

  // CF dialect ops are dynamically legal when their branch operands are either
  // register types or constants. The condition stays as i1.
  target.addDynamicallyLegalOp<cf::CondBranchOp>([&](cf::CondBranchOp op) {
    return cfOperandsLegal(op.getTrueDestOperands()) &&
           cfOperandsLegal(op.getFalseDestOperands());
  });
  target.addDynamicallyLegalOp<cf::BranchOp>(
      [&](cf::BranchOp op) { return cfOperandsLegal(op.getDestOperands()); });

  // KernelOp is dynamically legal - it becomes legal once the
  // KernelOpConversion pattern has converted all block argument types.
  // Start as illegal to ensure the pattern runs.
  target.addDynamicallyLegalOp<amdgcn::KernelOp>([&](amdgcn::KernelOp op) {
    // Check if any block in the body has non-register, non-i1 arguments
    for (Block &block : op.getBodyRegion()) {
      for (BlockArgument arg : block.getArguments()) {
        Type t = arg.getType();
        if (!isa<RegisterTypeInterface>(t) && !t.isInteger(1))
          return false;
      }
    }
    return true;
  });

  // Add the patterns.
  patterns.add<ArithBinaryOpPattern<arith::AddIOp, lsir::AddIOp>,
               ArithBinaryOpPattern<arith::SubIOp, lsir::SubIOp>,
               ArithBinaryOpPattern<arith::MulIOp, lsir::MulIOp>,
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
               ArithSelectOpPattern,
               // ASTER-specific abstractions used to connect pieces in
               // composable fashion.
               FromToRegOpPattern<ToRegOp>, FromToRegOpPattern<FromRegOp>,
               RegConstraintPattern, AssumeRangeOpPattern,
               // These patterns go together for proper composable control-flow
               // support. CF patterns need the type converter to handle block
               // argument conversion. KernelOp conversion handles block
               // argument types in kernel bodies. Cmp ops are converted to lsir
               // counterparts returning i1, which persists late in the pipeline
               // and is only translated to SCC after register allocation,
               // together with cf branch operations.
               ArithCmpIOpPattern, ArithCmpFOpPattern, CFCondBranchOpPattern,
               CFBranchOpPattern, KernelOpConversion, AssumeUniformOpPattern
               // That's all folks!
               >(converter);
  // Special generic pattern: converts operations by converting
  // their result types and recreating them.
  patterns.add<GenericOpConversion<RegConstraintOp>>(converter,
                                                     patterns.getContext());
}
