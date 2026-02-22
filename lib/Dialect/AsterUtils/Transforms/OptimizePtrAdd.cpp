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
#include "mlir/IR/AffineMap.h"
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

/// Represents the decomposed components of an offset expression.
struct OffsetComponents {
  /// Analyze the given offset value and decompose it into its components.
  static FailureOr<OffsetComponents> analyzeOffset(Value offset,
                                                   DataFlowSolver &solver);

  /// Get the affine map representing the offset expression.
  AffineMap getOffsetExpression() const { return offsetExpression; }

  /// Get the mapping from values to affine symbol positions.
  ArrayRef<std::pair<Value, int64_t>> getValueToAffinePos() const {
    return valueToAffinePos.getArrayRef();
  }

private:
  struct Offsets {
    AffineExpr constOffset;
    AffineExpr uniformOffset;
    AffineExpr dynamicOffset;

    Offsets(AffineExpr constOffset, AffineExpr uniformOffset,
            AffineExpr dynamicOffset)
        : constOffset(constOffset), uniformOffset(uniformOffset),
          dynamicOffset(dynamicOffset) {}

    /// Get the affine map representing the components.
    AffineMap getAsMap(int32_t numSyms, MLIRContext *context);

    /// Add another set of components to this one. The addition is done
    /// component-wise.
    void add(const Offsets &other);

    /// Multiply the components by another set of components. The multiplication
    /// is done distributively.
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
  AffineMap offsetExpression;
};
} // namespace

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

/// Returns whether the given value is a valid term for offset analysis.
/// A valid term is a non-negative integer value of bitwidth up to 64.
static FailureOr<unsigned> isValidTerm(Value value, DataFlowSolver &solver) {
  // Bail out if not an integer type.
  if (!value.getType().isSignlessInteger()) {
    LDBG() << "  Non-integer offset type in: " << value;
    return failure();
  }

  // Verify non-negativity.
  if (!succeeded(mlir::dataflow::staticallyNonNegative(solver, value))) {
    LDBG() << "  Non-positive offset in: " << value;
    return failure();
  }

  // Get the bitwidth of the integer type.
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();

  // Bail out on unsupported bitwidths.
  if (bitwidth > 64) {
    LDBG() << "  Unsupported bitwidth > 64 in: " << value;
    return failure();
  }

  return bitwidth;
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

AffineMap OffsetComponents::Offsets::getAsMap(int32_t numSyms,
                                              MLIRContext *context) {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/numSyms,
                        {constOffset, uniformOffset, dynamicOffset}, context);
}

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
  int64_t pos =
      valueToAffinePos
          .insert({value, static_cast<int64_t>(valueToAffinePos.size())})
          .first->second;
  return getAffineSymbolExpr(pos, context);
}

FailureOr<OffsetComponents>
OffsetComponents::analyzeOffset(Value offset, DataFlowSolver &solver) {
  OffsetComponents components(offset.getContext(), solver);
  FailureOr<Offsets> offExpr = components.analyzeTerm(offset);
  if (!succeeded(offExpr))
    return failure();
  components.offsetExpression = offExpr->getAsMap(
      components.valueToAffinePos.size(), offset.getContext());
  // Simplify the affine map.
  components.offsetExpression = simplifyAffineMap(components.offsetExpression);
  return components;
}

FailureOr<OffsetComponents::Offsets>
OffsetComponents::analyzeTerm(Value value) {
  // Helper lambda to get offsets for a value.
  auto getOffsets = [&](Value value, bool isKnownNonUniform = false) {
    if (!isKnownNonUniform && isUniform(value, solver)) {
      return Offsets(getAffineConstantExpr(0, context), getAsExpr(value),
                     getAffineConstantExpr(0, context));
    }
    return Offsets(getAffineConstantExpr(0, context),
                   getAffineConstantExpr(0, context), getAsExpr(value));
  };

  unsigned bitwidth = 0;
  // Get the bitwidth of the integer type.
  if (auto bitwidthOrErr = isValidTerm(value, solver);
      succeeded(bitwidthOrErr)) {
    bitwidth = *bitwidthOrErr;
  } else {
    return failure();
  }

  // Check if this is a constant.
  if (std::optional<APInt> constVal = getConstantValue(value, solver)) {
    assert(bitwidth == constVal->getBitWidth() && "bitwidth mismatch");
    return Offsets(getAffineConstantExpr(constVal->getSExtValue(), context),
                   getAffineConstantExpr(0, context),
                   getAffineConstantExpr(0, context));
  }

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
      AffineExpr cExpr = getAffineConstantExpr(1ULL << shift, context);
      AffineExpr zExpr = getAffineConstantExpr(0, context);
      lhs->mul(Offsets(cExpr, zExpr, zExpr));
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
  if (failed(componentsOrErr)) {
    // Analysis failed (e.g., offset not provably non-negative). Still convert
    // to aster_utils.ptr_add with the whole offset as the dynamic component.
    IRRewriter rewriter(op);
    auto constOffsetAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0);
    rewriter.replaceOpWithNewOp<aster_utils::PtrAddOp>(
        op, op.getResult().getType(), op.getBase(), offset,
        /*uniformOffset=*/nullptr, constOffsetAttr);
    return;
  }

  OffsetComponents &components = *componentsOrErr;
  AffineMap offsetExpr = components.getOffsetExpression();

  // The map has 3 results: [constOffset, uniformOffset, dynamicOffset]
  AffineExpr constExpr = offsetExpr.getResult(0);
  AffineExpr uniformExpr = offsetExpr.getResult(1);
  AffineExpr dynamicExpr = offsetExpr.getResult(2);

  // Check if the const expression is a constant.
  auto constCst = dyn_cast<AffineConstantExpr>(constExpr);
  if (!constCst)
    return;
  int64_t constOffsetVal = constCst.getValue();

  // Build the operands array from the value-to-position mapping.
  ArrayRef<std::pair<Value, int64_t>> valueToPos =
      components.getValueToAffinePos();
  SmallVector<Value> operands(valueToPos.size());
  for (auto [value, pos] : valueToPos)
    operands[pos] = value;

  IRRewriter rewriter(op);
  Location loc = op.getLoc();

  // Build the dynamic offset.
  Value dynamicOffset =
      materializeAffineExpr(rewriter, loc, dynamicExpr, offsetType, operands);

  // Build the uniform offset (optional).
  Value uniformOffset;
  if (!isa<AffineConstantExpr>(uniformExpr) ||
      cast<AffineConstantExpr>(uniformExpr).getValue() != 0)
    uniformOffset =
        materializeAffineExpr(rewriter, loc, uniformExpr, offsetType, operands);

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
