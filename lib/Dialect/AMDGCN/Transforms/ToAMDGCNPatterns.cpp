//===- ToAMDGCNPatterns.cpp
//--------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert to PIR patterns
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/ToAMDGCN.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

struct AddIOpPattern : public OpToAMDGCNPattern<pir::AddIOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

struct AllocaOpPattern : public OpToAMDGCNPattern<pir::AllocaOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AddIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AddIOpPattern::matchAndRewrite(Op op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Type uTy = op.getType().getType();
  // Skip unsupported bitwidths.
  if (uTy.getIntOrFloatBitWidth() > 64) {
    return rewriter.notifyMatchFailure(op, "unsupported bitwidth for pir.addi");
  }

  // Handle half-width addition.
  if (uTy.getIntOrFloatBitWidth() <= 16) {
    Value res = amdgcn::VAddU16::create(rewriter, op.getLoc(), adaptor.getDst(),
                                        adaptor.getLhs(), adaptor.getRhs())
                    .getDstRes();
    rewriter.replaceOp(op, res);
    return success();
  }

  // Handle single register addition.
  if (uTy.getIntOrFloatBitWidth() <= 32) {
    Value res = amdgcn::VAddU32::create(rewriter, op.getLoc(), adaptor.getDst(),
                                        adaptor.getLhs(), adaptor.getRhs())
                    .getDstRes();
    rewriter.replaceOp(op, res);
    return success();
  }

  // Handle larger widths by splitting into multiple registers.
  ValueRange dst = getOrSplitRange(rewriter, op.getLoc(), adaptor.getDst());
  ValueRange lhs = getOrSplitRange(rewriter, op.getLoc(), adaptor.getLhs());
  ValueRange rhs = getOrSplitRange(rewriter, op.getLoc(), adaptor.getRhs());
  assert(dst.size() == 2 && dst.size() == lhs.size() &&
         lhs.size() == rhs.size() && "Mismatched register range sizes");

  // Create the carry in and carry out registers.
  Value carryIn = createAllocation(
      rewriter, op.getLoc(),
      rewriter.getType<SGPRRangeType>(RegisterRange(Register(), 2)));
  Value carryOut = createAllocation(
      rewriter, op.getLoc(),
      rewriter.getType<SGPRRangeType>(RegisterRange(Register(), 2)));

  // Perform the low and high part additions.
  inst::VAddIOp loRes = amdgcn::VAddCoU32::create(rewriter, op.getLoc(), dst[0],
                                                  carryIn, lhs[0], rhs[0]);
  Value hiRes =
      amdgcn::VAddcCoU32::create(rewriter, op.getLoc(), dst[1], carryOut,
                                 lhs[1], rhs[1], loRes.getCarryOutRes())
          .getDstRes();
  // Replace the original operation.
  rewriter.replaceOp(op,
                     MakeRegisterRangeOp::create(rewriter, op.getLoc(),
                                                 {loRes.getDstRes(), hiRes}));
  return success();
}

//===----------------------------------------------------------------------===//
// AllocaOpPattern
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(Op op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value alloc =
      createAllocation(rewriter, op.getLoc(), converter.convertType(op));
  rewriter.replaceOp(op, alloc);
  return success();
}

//===----------------------------------------------------------------------===//
// ToAMDGCNPass patterns
//===----------------------------------------------------------------------===//

void mlir::aster::amdgcn::populateToAMDGCNPatterns(ToAMDGCNConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target) {
  // Configure the conversion target.
  target.addLegalDialect<amdgcn::AMDGCNDialect>();
  target.addIllegalDialect<pir::PIRDialect>();

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return op.getType().isIntOrIndexOrFloat(); });

  // Add the patterns.
  patterns.add<AddIOpPattern, AllocaOpPattern>(converter);
}
