//===- LegalizeOperands.cpp - Legalize operand constraints ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes operand constraints that cannot be satisfied by the
// hardware encoding. For example, SOP2 instructions allow at most one literal
// (non-inline) constant operand. If an lsir.select has two non-inline literal
// operands, this pass materializes one into a register via alloca + s_mov_b32.
//
// This pass runs before bufferization and register allocation so that the
// newly created allocas participate in the normal register allocation flow.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LEGALIZEOPERANDS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

/// Return the integer value of an arith.constant, or std::nullopt.
static std::optional<int64_t> getConstInt(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
      return intAttr.getInt();
  return std::nullopt;
}

/// AMD GCN inline integer constants are in [-16, 64].
static bool isInlineInt(int64_t val) { return val >= -16 && val <= 64; }

struct LegalizeOperands
    : public amdgcn::impl::LegalizeOperandsBase<LegalizeOperands> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void LegalizeOperands::runOnOperation() {
  Operation *op = getOperation();

  SmallVector<lsir::SelectOp> toFix;
  op->walk([&](lsir::SelectOp selectOp) {
    auto trueConst = getConstInt(selectOp.getTrueValue());
    auto falseConst = getConstInt(selectOp.getFalseValue());
    if (trueConst && falseConst && !isInlineInt(*trueConst) &&
        !isInlineInt(*falseConst))
      toFix.push_back(selectOp);
  });

  for (lsir::SelectOp selectOp : toFix) {
    IRRewriter rewriter(selectOp);
    rewriter.setInsertionPoint(selectOp);
    Location loc = selectOp.getLoc();

    // Materialize the true_value literal into a new sgpr via alloca +
    // s_mov_b32.
    Type sgprTy = SGPRType::get(rewriter.getContext(), Register());
    Value out = AllocaOp::create(rewriter, loc, sgprTy);
    auto instAttr = InstAttr::get(rewriter.getContext(), OpCode::S_MOV_B32);
    Value movResult = inst::SOP1Op::create(rewriter, loc, sgprTy, instAttr, out,
                                           selectOp.getTrueValue())
                          .getSdstRes();

    // Replace the true_value operand with the register value.
    selectOp.getTrueValueMutable().assign(movResult);
  }
}
