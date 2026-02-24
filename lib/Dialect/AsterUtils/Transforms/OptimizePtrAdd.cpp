//===- OptimizePtrAdd.cpp - Optimize ptr.ptr_add operations ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "optimize-ptr-add"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_OPTIMIZEPTRADD
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// OptimizePtrAdd pass
//===----------------------------------------------------------------------===//

struct OptimizePtrAdd
    : public aster_utils::impl::OptimizePtrAddBase<OptimizePtrAdd> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// OffsetComponents
//===----------------------------------------------------------------------===//

/// Represents the decomposed (const, uniform, dynamic) components of an offset
/// expression after analysis. constPart is always an AffineConstantExpr.
/// operands[i] is the Value corresponding to affine symbol s_i.
struct OffsetComponents {
  AffineExpr constPart;
  AffineExpr uniformPart;
  AffineExpr dynamicPart;
  SmallVector<Value> operands;

  /// Analyze the given offset value and decompose it into its components.
  static FailureOr<OffsetComponents> analyzeOffset(Value offset,
                                                   DataFlowSolver &solver);

private:
  struct Offsets {
    AffineExpr constOffset;
    AffineExpr uniformOffset;
    AffineExpr dynamicOffset;

    /// Named constructors for common cases.
    static Offsets cst(AffineExpr expr, MLIRContext *ctx) {
      auto z = getAffineConstantExpr(0, ctx);
      return {expr, z, z};
    }
    static Offsets uniform(AffineExpr expr, MLIRContext *ctx) {
      auto z = getAffineConstantExpr(0, ctx);
      return {z, expr, z};
    }
    static Offsets dynamic(AffineExpr expr, MLIRContext *ctx) {
      auto z = getAffineConstantExpr(0, ctx);
      return {z, z, expr};
    }

    // Component-wise addition.
    void add(const Offsets &other);

    // Cross-component multiplication.
    void mul(const Offsets &other);
  };

  OffsetComponents(MLIRContext *ctx, DataFlowSolver &solver)
      : context(ctx), solver(solver) {}

  /// Recursively analyze an additive expression. It is assumed that
  /// `multiplier` is always a constant or uniform value.
  FailureOr<Offsets> analyzeTerm(Value value);
  /// Get the affine expression for the given value.
  AffineExpr getAsExpr(Value value);

  MLIRContext *context;
  DataFlowSolver &solver;
  llvm::MapVector<Value, int64_t> valueToAffinePos;
};
} // namespace

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

/// Returns whether the given value is a valid term for offset analysis.
/// A valid term is a non-negative signless integer value of bitwidth <= 64.
static bool isValidTerm(Value value, DataFlowSolver &solver) {
  if (!value.getType().isSignlessInteger() ||
      value.getType().getIntOrFloatBitWidth() > 64) {
    LDBG() << "  Invalid offset type: " << value;
    return false;
  }
  if (!succeeded(mlir::dataflow::staticallyNonNegative(solver, value))) {
    LDBG() << "  Non-positive offset: " << value;
    return false;
  }
  return true;
}

/// Returns the constant value of the given value if it can be determined.
static std::optional<APInt> getConstantValue(Value value,
                                             DataFlowSolver &solver) {
  auto *inferredRange =
      solver.lookupState<mlir::dataflow::IntegerValueRangeLattice>(value);
  if (!inferredRange || inferredRange->getValue().isUninitialized())
    return std::nullopt;
  return inferredRange->getValue().getValue().getConstantValue();
}

/// Returns whether the given value is uniform across threads.
static bool isUniform(Value value, DataFlowSolver &solver) {
  auto *lattice =
      solver.lookupState<aster::dataflow::ThreadUniformLattice>(value);
  return lattice && lattice->getValue().isUniform();
}

//===----------------------------------------------------------------------===//
// OffsetComponents::Offsets
//===----------------------------------------------------------------------===//

void OffsetComponents::Offsets::add(const Offsets &other) {
  constOffset = constOffset + other.constOffset;
  uniformOffset = uniformOffset + other.uniformOffset;
  dynamicOffset = dynamicOffset + other.dynamicOffset;
}

void OffsetComponents::Offsets::mul(const Offsets &other) {
  // Distributive multiplication:
  // (c1 + u1 + d1) * (c2 + u2 + d2)
  AffineExpr newConst = constOffset * other.constOffset;
  AffineExpr newUniform = constOffset * other.uniformOffset +
                          uniformOffset * other.constOffset +
                          uniformOffset * other.uniformOffset;
  AffineExpr newDynamic = dynamicOffset * other.constOffset +
                          dynamicOffset * other.uniformOffset +
                          dynamicOffset * other.dynamicOffset;

  constOffset = newConst;
  uniformOffset = newUniform;
  dynamicOffset = newDynamic;
}

//===----------------------------------------------------------------------===//
// OffsetComponents
//===----------------------------------------------------------------------===//

AffineExpr OffsetComponents::getAsExpr(Value value) {
  // If constant, return constant expression.
  if (std::optional<APInt> constVal = getConstantValue(value, solver))
    return getAffineConstantExpr(constVal->getSExtValue(), context);

  // Get a position for the affine symbol.
  auto it = valueToAffinePos.insert(
      {value, static_cast<int64_t>(valueToAffinePos.size())});
  int64_t pos = it.first->second;
  return getAffineSymbolExpr(pos, context);
}

FailureOr<OffsetComponents>
OffsetComponents::analyzeOffset(Value offset, DataFlowSolver &solver) {
  OffsetComponents components(offset.getContext(), solver);
  FailureOr<Offsets> offExpr = components.analyzeTerm(offset);
  if (failed(offExpr))
    return failure();

  int32_t numSyms = components.valueToAffinePos.size();
  // Simplify each component expression directly.
  components.constPart = simplifyAffineExpr(offExpr->constOffset, 0, numSyms);
  components.uniformPart =
      simplifyAffineExpr(offExpr->uniformOffset, 0, numSyms);
  components.dynamicPart =
      simplifyAffineExpr(offExpr->dynamicOffset, 0, numSyms);
  // Build the operands array ordered by affine symbol position.
  components.operands.resize(numSyms);
  for (auto [value, pos] : components.valueToAffinePos)
    components.operands[pos] = value;
  return components;
}

FailureOr<OffsetComponents::Offsets>
OffsetComponents::analyzeTerm(Value value) {
  // Classify a leaf value as uniform or dynamic.
  auto getOffsets = [&](Value value, bool isKnownNonUniform = false) {
    if (!isKnownNonUniform && isUniform(value, solver))
      return Offsets::uniform(getAsExpr(value), context);
    return Offsets::dynamic(getAsExpr(value), context);
  };

  if (!isValidTerm(value, solver))
    return failure();

  // Check if this is a constant.
  if (std::optional<APInt> constVal = getConstantValue(value, solver))
    return Offsets::cst(
        getAffineConstantExpr(constVal->getSExtValue(), context), context);

  auto asResult = dyn_cast<OpResult>(value);
  // Handle values not defined by an operation.
  if (!asResult)
    return getOffsets(value);

  Operation *defOp = asResult.getOwner();

  // Bail if there are no overflow flags. Since we expect non-negative offsets,
  // it's safe to assume that nsw implies nuw.
  if (auto aOp = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(defOp);
      !aOp || (!aOp.hasNoSignedWrap() && !aOp.hasNoUnsignedWrap())) {
    LDBG() << "  Cannot decompose value due to invalid overflow flags: "
           << value;
    return getOffsets(value);
  }

  // Handle additive operations.
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    FailureOr<Offsets> lhs = analyzeTerm(addOp.getLhs());
    // If the left-hand side analysis failed, bail out and treat the add as a
    // single term.
    if (failed(lhs))
      return getOffsets(addOp);

    FailureOr<Offsets> rhs = analyzeTerm(addOp.getRhs());
    // If the right-hand side analysis failed, bail out and treat the add as a
    // single term.
    if (failed(rhs))
      return getOffsets(addOp);

    lhs->add(*rhs);
    return *lhs;
  }

  // Handle multiplicative operations.
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    // If the multiplication result is not uniform, bail out as we can't split.
    if (!isUniform(mulOp, solver))
      return getOffsets(mulOp, true);

    FailureOr<Offsets> lhs = analyzeTerm(mulOp.getLhs());
    // If the left-hand side analysis failed, bail out and treat the mul as a
    // single term.
    if (failed(lhs))
      return getOffsets(mulOp);

    FailureOr<Offsets> rhs = analyzeTerm(mulOp.getRhs());
    // If the right-hand side analysis failed, bail out and treat the mul as a
    // single term.
    if (failed(rhs))
      return getOffsets(mulOp);

    lhs->mul(*rhs);
    return *lhs;
  }

  // Handle shift left operations.
  if (auto shlOp = dyn_cast<arith::ShLIOp>(defOp)) {
    // We can only handle constant shift amounts.
    if (auto shiftAmt = getConstantValue(shlOp.getRhs(), solver)) {
      int64_t shift = shiftAmt->getSExtValue();
      FailureOr<Offsets> lhs = analyzeTerm(shlOp.getLhs());
      if (failed(lhs))
        return getOffsets(shlOp);
      lhs->mul(
          Offsets::cst(getAffineConstantExpr(1ULL << shift, context), context));
      return *lhs;
    }
  }

  // Handle assume_range operations.
  if (auto assumeOp = dyn_cast<aster_utils::AssumeRangeOp>(defOp))
    return analyzeTerm(assumeOp.getInput());

  return getOffsets(value);
}

//===----------------------------------------------------------------------===//
// Transform
//===----------------------------------------------------------------------===//

static Value materializeAffineExpr(IRRewriter &rewriter, Location loc,
                                   AffineExpr expr, Type resultType,
                                   ArrayRef<Value> operands) {
  // Handle constant expression.
  if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
    return arith::ConstantIntOp::create(rewriter, loc, resultType,
                                        cst.getValue());
  }

  // Handle symbol expression.
  if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
    return operands[sym.getPosition()];

  // Handle binary expressions.
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    Value lhs = materializeAffineExpr(rewriter, loc, binExpr.getLHS(),
                                      resultType, operands);
    Value rhs = materializeAffineExpr(rewriter, loc, binExpr.getRHS(),
                                      resultType, operands);

    arith::IntegerOverflowFlags flags =
        arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw;
    switch (binExpr.getKind()) {
    case AffineExprKind::Add:
      return arith::AddIOp::create(rewriter, loc, lhs, rhs, flags);
    case AffineExprKind::Mul:
      return arith::MulIOp::create(rewriter, loc, lhs, rhs, flags);
    case AffineExprKind::FloorDiv:
      return arith::DivSIOp::create(rewriter, loc, lhs, rhs);
    case AffineExprKind::CeilDiv:
      return arith::CeilDivSIOp::create(rewriter, loc, lhs, rhs);
    case AffineExprKind::Mod:
      return arith::RemSIOp::create(rewriter, loc, lhs, rhs);
    default:
      llvm_unreachable("unexpected affine expr kind");
    }
  }

  llvm_unreachable("unexpected affine expr");
}

static void optimizePtrAddOp(ptr::PtrAddOp op, DataFlowSolver &solver) {
  if (op.getFlags() == ptr::PtrAddFlags::none)
    return;

  Value offset = op.getOffset();
  Type offsetType = offset.getType();

  // Analyze the offset expression.
  FailureOr<OffsetComponents> componentsOrErr =
      OffsetComponents::analyzeOffset(offset, solver);
  if (failed(componentsOrErr))
    return;

  OffsetComponents &components = *componentsOrErr;

  // Check if the const component is a compile-time constant.
  auto constCst = dyn_cast<AffineConstantExpr>(components.constPart);
  if (!constCst)
    return;
  int64_t constOffsetVal = constCst.getValue();

  IRRewriter rewriter(op);
  Location loc = op.getLoc();

  // Build the dynamic offset.
  Value dynamicOffset = materializeAffineExpr(
      rewriter, loc, components.dynamicPart, offsetType, components.operands);

  // Build the uniform offset (optional).
  Value uniformOffset = nullptr;
  if (!isa<AffineConstantExpr>(components.uniformPart) ||
      cast<AffineConstantExpr>(components.uniformPart).getValue() != 0)
    uniformOffset = materializeAffineExpr(rewriter, loc, components.uniformPart,
                                          offsetType, components.operands);

  // Create the optimized ptr_add operation.
  auto constOffsetAttr =
      rewriter.getIntegerAttr(rewriter.getI64Type(), constOffsetVal);
  rewriter.replaceOpWithNewOp<aster_utils::PtrAddOp>(
      op, op.getResult().getType(), op.getBase(), dynamicOffset, uniformOffset,
      constOffsetAttr);
}

//===----------------------------------------------------------------------===//
// OptimizePtrAdd
//===----------------------------------------------------------------------===//

void OptimizePtrAdd::runOnOperation() {
  Operation *op = getOperation();

  // Set up the data flow solver with required analyses.
  DataFlowSolver solver;
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<mlir::dataflow::IntegerRangeAnalysis>();
  solver.load<aster::dataflow::ThreadUniformAnalysis>();

  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  op->walk([&](ptr::PtrAddOp ptrAddOp) { optimizePtrAddOp(ptrAddOp, solver); });
}
