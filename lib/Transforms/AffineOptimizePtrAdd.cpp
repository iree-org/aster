//===- AffineOptimizePtrAdd.cpp - Optimize ptr_add with affine_apply ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes ptr.ptr_add operations when the offset is produced by
// affine.apply. It decomposes the affine expression into const, uniform, and
// dynamic components using ValueBounds (instead of IntRange) for constant
// detection.
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "affine-optimize-ptr-add"

namespace mlir::aster {
#define GEN_PASS_DEF_AFFINEOPTIMIZEPTRADD
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// AffineOptimizePtrAdd pass
//===----------------------------------------------------------------------===//

struct AffineOptimizePtrAdd
    : public aster::impl::AffineOptimizePtrAddBase<AffineOptimizePtrAdd> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Offsets struct
//===----------------------------------------------------------------------===//

/// Represents the decomposed (const, uniform, dynamic) components of an offset.
struct Offsets {
  AffineExpr constOffset;
  AffineExpr uniformOffset;
  AffineExpr dynamicOffset;

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

  void add(const Offsets &other) {
    constOffset = constOffset + other.constOffset;
    uniformOffset = uniformOffset + other.uniformOffset;
    dynamicOffset = dynamicOffset + other.dynamicOffset;
  }

  void mul(const Offsets &other) {
    AffineExpr newConst = constOffset * other.constOffset;
    AffineExpr newUniform = constOffset * other.uniformOffset +
                            uniformOffset * other.constOffset +
                            uniformOffset * other.uniformOffset;
    AffineExpr newDynamic =
        dynamicOffset *
            (other.constOffset + other.uniformOffset + other.dynamicOffset) +
        other.dynamicOffset * (constOffset + uniformOffset);
    constOffset = newConst;
    uniformOffset = newUniform;
    dynamicOffset = newDynamic;
  }
};

//===----------------------------------------------------------------------===//
// OffsetTransformImpl
//===----------------------------------------------------------------------===//

/// Classification of each operand: constant, uniform, or dynamic.
enum class TermClass { Constant, Uniform, Dynamic };

/// Implements the offset transformation algorithm.
struct OffsetTransformImpl {
  /// Analyzes the given affine expression and returns the decomposed (const,
  /// uniform, dynamic) components.
  static FailureOr<Offsets> analyzeAffineApply(AffineExpr expr,
                                               ArrayRef<bool> isUniform,
                                               ValueRange operands,
                                               bool assumePositive);

private:
  /// Analyzes the given term and returns the decomposed (const, uniform,
  /// dynamic) components.
  Offsets analyzeTerm(AffineExpr expr);

  /// Returns true if the given expression is dynamic.
  bool isDynamic(AffineExpr expr) const;

  OffsetTransformImpl(MLIRContext *ctx, ArrayRef<bool> isUniform)
      : context(ctx), isUniform(isUniform) {}
  MLIRContext *context = nullptr;
  ArrayRef<bool> isUniform;
  SmallVector<unsigned> dynamicSymbols;
};
} // namespace

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Returns whether the given value is uniform across threads.
static bool isUniform(Value value, DataFlowSolver &solver) {
  auto *lattice =
      solver.lookupState<aster::dataflow::ThreadUniformLattice>(value);
  return lattice && lattice->getValue().isUniform();
}

//===----------------------------------------------------------------------===//
// OffsetTransformImpl::analyzeAffineApply
//===----------------------------------------------------------------------===//

FailureOr<Offsets> OffsetTransformImpl::analyzeAffineApply(
    AffineExpr expr, ArrayRef<bool> isUniform, ValueRange operands,
    bool assumePositive) {
  OffsetTransformImpl impl{expr.getContext(), isUniform};

  // Collect the dynamic symbols.
  impl.dynamicSymbols.reserve(llvm::count(isUniform, false));
  for (auto [i, uniform] : llvm::enumerate(isUniform)) {
    if (uniform)
      continue;
    impl.dynamicSymbols.push_back(i);
  }

  Offsets offsets = impl.analyzeTerm(expr);
  auto zerorAttr = IntegerAttr::get(IndexType::get(expr.getContext()), 0);

  // Helper to check if the given expression is non-negative.
  auto isPositive = [&](AffineExpr expr) {
    AffineMap map =
        AffineMap::get(0, operands.size(), {expr}, expr.getContext());
    ValueBoundsConstraintSet::Variable lhs(
        OpFoldResult(cast<Attribute>(zerorAttr)));
    ValueBoundsConstraintSet::Variable rhs(map, operands);
    LDBG() << "  Comparing: " << lhs.getMap() << " <= " << rhs.getMap();
    return ValueBoundsConstraintSet::compare(
        lhs, ValueBoundsConstraintSet::ComparisonOperator::LE, rhs);
  };

  // Simplify the offset expressions.
  offsets.constOffset =
      simplifyAffineExpr(offsets.constOffset, 0, operands.size());
  offsets.uniformOffset =
      simplifyAffineExpr(offsets.uniformOffset, 0, operands.size());
  offsets.dynamicOffset =
      simplifyAffineExpr(offsets.dynamicOffset, 0, operands.size());

  // Check if the offset expressions are non-negative.
  LDBG() << "  Const offset: " << offsets.constOffset;
  if (!assumePositive && !isPositive(offsets.constOffset)) {
    LDBG() << "  Constant offset is not non-negative: " << offsets.constOffset;
    return failure();
  }
  LDBG() << "  Uniform offset: " << offsets.uniformOffset;
  if (!assumePositive && !isPositive(offsets.uniformOffset)) {
    LDBG() << "  Uniform offset is not non-negative: " << offsets.uniformOffset;
    return failure();
  }
  LDBG() << "  Dynamic offset: " << offsets.dynamicOffset;
  if (!assumePositive && !isPositive(offsets.dynamicOffset)) {
    LDBG() << "  Dynamic offset is not non-negative: " << offsets.dynamicOffset;
    return failure();
  }
  return offsets;
}

bool OffsetTransformImpl::isDynamic(AffineExpr expr) const {
  return llvm::any_of(dynamicSymbols, [expr](unsigned pos) {
    return expr.isFunctionOfSymbol(pos);
  });
}

Offsets OffsetTransformImpl::analyzeTerm(AffineExpr expr) {
  // Handle constant expression.
  if (auto cst = dyn_cast<AffineConstantExpr>(expr))
    return Offsets::cst(cst, context);

  // Handle symbol expression.
  if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
    int64_t pos = sym.getPosition();
    if (isUniform[pos])
      return Offsets::uniform(sym, context);
    return Offsets::dynamic(sym, context);
  }

  AffineExprKind kind = expr.getKind();
  // If the expression is not an add or mul, treat it as a single term.
  if (kind != AffineExprKind::Add && kind != AffineExprKind::Mul) {
    if (isDynamic(expr))
      return Offsets::dynamic(expr, context);
    return Offsets::uniform(expr, context);
  }

  auto binExpr = cast<AffineBinaryOpExpr>(expr);
  Offsets lhs = analyzeTerm(binExpr.getLHS());
  Offsets rhs = analyzeTerm(binExpr.getRHS());
  if (kind == AffineExprKind::Add)
    lhs.add(rhs);
  else
    lhs.mul(rhs);
  return lhs;
}

//===----------------------------------------------------------------------===//
// Materialize and transform
//===----------------------------------------------------------------------===//

/// Optimizes ptr_add when offset is from affine_apply.
static void optimizePtrAddWithAffineOffset(IRRewriter &rewriter,
                                           ptr::PtrAddOp op,
                                           affine::AffineApplyOp applyOp,
                                           DataFlowSolver &solver,
                                           bool assumePositive) {
  // Simplify and canonicalize the affine map.
  AffineMap map = applyOp.getAffineMap();
  if (map.getNumResults() != 1)
    return;
  SmallVector<Value> operands;
  llvm::append_range(operands, applyOp.getMapOperands());
  affine::fullyComposeAffineMapAndOperands(&map, &operands, true);
  affine::canonicalizeMapAndOperands(&map, &operands);

  // Make a new map where the dimensions are now the leading symbols.
  {
    SmallVector<AffineExpr> dimReplacements(map.getNumDims());
    SmallVector<AffineExpr> symReplacements(map.getNumSymbols());
    int64_t numDims = map.getNumDims();
    for (auto &&[i, e] : llvm::enumerate(dimReplacements))
      e = getAffineSymbolExpr(i, applyOp.getContext());
    for (auto &&[i, e] : llvm::enumerate(symReplacements))
      e = getAffineSymbolExpr(i + numDims, applyOp.getContext());
    map.replaceDimsAndSymbols(dimReplacements, symReplacements, 0,
                              operands.size());
  }

  // Simplify the affine map.
  map = simplifyAffineMap(map);

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Optimizing op: " << op << "\n  Affine map: " << map;
    os << "\n  Affine operands: " << llvm::interleaved_array(operands);
    os << "\n  Apply op: " << applyOp;
  });

  // Get whether each operand is uniform.
  SmallVector<bool> isUniformValue(operands.size(), false);
  for (auto &&[i, e] : llvm::enumerate(isUniformValue))
    e = isUniform(operands[i], solver);

  LDBG() << "  Is uniform: " << llvm::interleaved_array(isUniformValue);

  // Analyze the offset expression.
  FailureOr<Offsets> componentsOrErr = OffsetTransformImpl::analyzeAffineApply(
      map.getResult(0), isUniformValue, operands, assumePositive);
  if (failed(componentsOrErr))
    return;

  Offsets &components = *componentsOrErr;

  LDBG() << "  Final components: " << components.constOffset << ", "
         << components.uniformOffset << ", " << components.dynamicOffset;

  // Get the constant offset.
  int64_t constOff = 0;
  if (auto constCst =
          dyn_cast_or_null<AffineConstantExpr>(components.constOffset))
    constOff = constCst.getValue();

  Location loc = op.getLoc();
  SmallVector<OpFoldResult> ofrs;
  ofrs = llvm::to_vector_of<OpFoldResult>(operands);

  // Build the dynamic offset.
  Value dynamicOffset;
  if (components.dynamicOffset) {
    dynamicOffset = affine::makeComposedAffineApply(
        rewriter, loc, components.dynamicOffset, ofrs);
  } else {
    dynamicOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  // Build the uniform offset.
  Value uniformOffset = nullptr;
  if (components.uniformOffset) {
    uniformOffset = affine::makeComposedAffineApply(
        rewriter, loc, components.uniformOffset, ofrs);
  }

  // Create the optimized ptr_add operation.
  Value result = op.getBase();
  if (uniformOffset) {
    result = ptr::PtrAddOp::create(rewriter, op.getLoc(), result, uniformOffset,
                                   op.getFlags());
  }
  if (dynamicOffset) {
    result = ptr::PtrAddOp::create(rewriter, op.getLoc(), result, dynamicOffset,
                                   op.getFlags());
  }
  if (constOff != 0) {
    auto cOp = arith::ConstantIndexOp::create(rewriter, op.getLoc(), constOff);
    result = ptr::PtrAddOp::create(rewriter, op.getLoc(), result, cOp,
                                   op.getFlags());
  }
  rewriter.replaceOp(op, result);
}

//===----------------------------------------------------------------------===//
// AffineOptimizePtrAdd
//===----------------------------------------------------------------------===//

void AffineOptimizePtrAdd::runOnOperation() {
  Operation *op = getOperation();

  // Configure and run the data flow solver.
  DataFlowSolver solver;
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<aster::dataflow::ThreadUniformAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  // Walk through the operations and optimize ptr_add with affine_apply offset.
  IRRewriter rewriter(op);
  op->walk([&](ptr::PtrAddOp ptrAddOp) {
    auto applyOp = ptrAddOp.getOffset().getDefiningOp<affine::AffineApplyOp>();
    if (!applyOp)
      return;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ptrAddOp);
    optimizePtrAddWithAffineOffset(rewriter, ptrAddOp, applyOp, solver,
                                   assumePositive);
  });
}
