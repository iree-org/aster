//===- FactorizeAffineExpr.cpp --------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Factorizes affine expressions to minimize multiplications by expanding to a
// sum-of-monomials polynomial, greedily extracting the most-frequent variable,
// and post-processing to extract common multiplicative factors across addends.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"
#include "aster/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster {
#define GEN_PASS_DEF_FACTORIZEAFFINEEXPR
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::affine;

namespace {
//===----------------------------------------------------------------------===//
// FactorizeAffineExpr pass
//===----------------------------------------------------------------------===//

struct FactorizeAffineExpr
    : public aster::impl::FactorizeAffineExprBase<FactorizeAffineExpr> {
  using Base::Base;

  void runOnOperation() override;
};
//===----------------------------------------------------------------------===//
// Polynomial representation
//===----------------------------------------------------------------------===//
// A monomial: coefficient * product of atomic AffineExprs.
// Atomic means: DimId, SymbolId, Constant, or a Mod/FloorDiv/CeilDiv expr.
struct Mono {
  int64_t coeff = 0;
  SmallVector<AffineExpr, 4> vars;
};

using Poly = SmallVector<Mono, 8>;
} // namespace

//===----------------------------------------------------------------------===//
// Canonical comparison for deterministic ordering
//===----------------------------------------------------------------------===//

/// Three-way comparison of two Affine expressions.
static int compareExpr(AffineExpr a, AffineExpr b) {
  if (a == b)
    return 0;

  // Compare the kinds of the expressions.
  AffineExprKind ka = a.getKind(), kb = b.getKind();
  if (ka != kb)
    return static_cast<int>(ka) < static_cast<int>(kb) ? -1 : 1;

  // Compare the values of the expressions.
  switch (ka) {
  case AffineExprKind::Constant: {
    // Compare the values of the constant expressions.
    int64_t va = cast<AffineConstantExpr>(a).getValue();
    int64_t vb = cast<AffineConstantExpr>(b).getValue();
    return va < vb ? -1 : (va > vb ? 1 : 0);
  }
  case AffineExprKind::DimId: {
    // Compare the positions of the dimension expressions.
    unsigned pa = cast<AffineDimExpr>(a).getPosition();
    unsigned pb = cast<AffineDimExpr>(b).getPosition();
    return pa < pb ? -1 : (pa > pb ? 1 : 0);
  }
  case AffineExprKind::SymbolId: {
    // Compare the positions of the symbol expressions.
    unsigned pa = cast<AffineSymbolExpr>(a).getPosition();
    unsigned pb = cast<AffineSymbolExpr>(b).getPosition();
    return pa < pb ? -1 : (pa > pb ? 1 : 0);
  }
  default:
    break;
  }

  // Compare the children of the binary expressions.
  auto ba = cast<AffineBinaryOpExpr>(a), bb = cast<AffineBinaryOpExpr>(b);
  int lc = compareExpr(ba.getLHS(), bb.getLHS());
  if (lc != 0)
    return lc;
  return compareExpr(ba.getRHS(), bb.getRHS());
}

//===----------------------------------------------------------------------===//
// Polynomial conversion
//===----------------------------------------------------------------------===//

/// Convert an Affine expression into a polynomial.
/// Mod/FloorDiv/CeilDiv are treated as atomic variables.
static void affineToPoly(AffineExpr expr, Poly &poly) {
  if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
    poly.push_back({cst.getValue(), {}});
    return;
  }
  if (isa<AffineDimExpr, AffineSymbolExpr>(expr)) {
    poly.push_back({1, {expr}});
    return;
  }
  auto binOp = cast<AffineBinaryOpExpr>(expr);
  AffineExprKind kind = binOp.getKind();
  if (kind != AffineExprKind::Add && kind != AffineExprKind::Mul) {
    poly.push_back({1, {expr}}); // atomic
    return;
  }
  if (kind == AffineExprKind::Add) {
    affineToPoly(binOp.getLHS(), poly);
    affineToPoly(binOp.getRHS(), poly);
    return;
  }
  // Mul: distribute over addition via cross-product.
  Poly lhsPoly, rhsPoly;
  affineToPoly(binOp.getLHS(), lhsPoly);
  affineToPoly(binOp.getRHS(), rhsPoly);

  // Distribute over addition via cross-product.
  for (const Mono &ma : lhsPoly) {
    for (const Mono &mb : rhsPoly) {
      Mono m;
      m.coeff = ma.coeff * mb.coeff;
      llvm::append_range(m.vars, ma.vars);
      llvm::append_range(m.vars, mb.vars);
      poly.push_back(std::move(m));
    }
  }
}

/// Convert a polynomial into an Affine expression.
static AffineExpr polyToAffine(const Poly &poly, MLIRContext *ctx) {
  AffineExpr result = getAffineConstantExpr(0, ctx);
  for (const Mono &m : poly) {
    result = result + (getAffineConstantExpr(m.coeff, ctx) *
                       computeProduct(ctx, m.vars));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Polynomial normalization
//===----------------------------------------------------------------------===//

/// Sort vars within each mono, sort monomials lexicographically, combine like
/// terms, drop zeros.
static void normalize(Poly &poly) {
  auto affineLess =
      +[](AffineExpr a, AffineExpr b) { return compareExpr(a, b) < 0; };

  // Sort vars within each mono.
  for (Mono &m : poly)
    llvm::sort(m.vars, affineLess);

  // Sort monomials lexicographically.
  auto monoLess = [&affineLess](const Mono &a, const Mono &b) {
    if (a.vars.size() != b.vars.size())
      return a.vars.size() < b.vars.size();
    return std::lexicographical_compare(
        a.vars.begin(), a.vars.end(), b.vars.begin(), b.vars.end(), affineLess);
  };
  llvm::sort(poly, monoLess);

  // Combine terms in place.
  int64_t curr = 0;
  for (int64_t term = 1, e = poly.size(); term < e; term++) {
    if (poly[curr].vars == poly[term].vars) {
      poly[curr].coeff += poly[term].coeff;
      poly[term].coeff = 0;
      continue;
    }
    curr++;
    if (curr != term)
      poly[curr] = std::move(poly[term]);
  }
  poly.resize(curr + 1);
  llvm::erase_if(poly, [](const Mono &m) { return m.coeff == 0; });
}

//===----------------------------------------------------------------------===//
// Greedy polynomial factorization
//===----------------------------------------------------------------------===//

/// Greedily factorize the polynomial by extracting the most frequent variable.
static AffineExpr greedyFactor(Poly poly, MLIRContext *ctx) {
  normalize(poly);
  if (poly.empty())
    return getAffineConstantExpr(0, ctx);

  // Get the most frequent variable.
  llvm::SmallDenseMap<AffineExpr, int> freqs;
  llvm::SmallDenseSet<AffineExpr, 8> seen;
  AffineExpr var;
  int mostFreq = 1;
  for (const Mono &m : poly) {
    seen.clear();
    for (AffineExpr v : m.vars) {
      if (!seen.insert(v).second)
        continue;
      int freq = ++freqs[v];
      if (freq > mostFreq) {
        mostFreq = freq;
        var = v;
      }
    }
  }

  if (!var)
    return polyToAffine(poly, ctx);

  // Split the polynominal
  Poly withoutPoly;
  for (Mono &m : poly) {
    auto it = llvm::find(m.vars, var);
    // If the variable is not found, add the monomial to the withoutPoly.
    if (it == m.vars.end()) {
      withoutPoly.push_back(std::move(m));

      // Set the coefficient to 0 to remove it from the polynomial.
      m.coeff = 0;
      continue;
    }
    m.vars.erase(it);
  }
  llvm::erase_if(poly, [](const Mono &m) { return m.coeff == 0; });

  // Recursively factorize the polynomial without the most frequent variable.
  AffineExpr product = greedyFactor(poly, ctx) * var;
  if (withoutPoly.empty())
    return product;

  // Process the polynomial without the most frequent variable.
  return product + greedyFactor(withoutPoly, ctx);
}

//===----------------------------------------------------------------------===//
// Common-factor post-processing
//===----------------------------------------------------------------------===//

/// Flatten a binary expression of the given kind into a list of expressions.
static void flattenBinary(AffineExpr expr, AffineExprKind kind,
                          SmallVectorImpl<AffineExpr> &out) {
  auto binOp = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binOp || binOp.getKind() != kind) {
    out.push_back(expr);
    return;
  }
  flattenBinary(binOp.getLHS(), kind, out);
  flattenBinary(binOp.getRHS(), kind, out);
}

// Remove one occurrence of `factor` from the product represented by `expr`.
static AffineExpr divideByFactor(AffineExpr expr, AffineExpr factor) {
  SmallVector<AffineExpr> factors;
  flattenBinary(expr, AffineExprKind::Mul, factors);
  auto it = llvm::find(factors, factor);
  assert(it != factors.end() && "factor not found in product");
  factors.erase(it);
  return computeProduct(expr.getContext(), factors);
}

// Extract common multiplicative factors from all addends of each Add node,
// then recurse. For non-Add nodes, recurse into sub-expressions.
static AffineExpr applyCommonFactors(AffineExpr expr) {
  auto binOp = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binOp)
    return expr;

  AffineExprKind kind = binOp.getKind();
  if (kind != AffineExprKind::Add) {
    AffineExpr lhs = applyCommonFactors(binOp.getLHS());
    AffineExpr rhs = applyCommonFactors(binOp.getRHS());
    switch (kind) {
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::Mod:
      return lhs % rhs;
    case AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      llvm_unreachable("unexpected AffineExprKind");
    }
  }

  // Flatten addends, recurse into each, then find common factors.
  SmallVector<AffineExpr> addends;
  flattenBinary(expr, AffineExprKind::Add, addends);
  for (AffineExpr &a : addends)
    a = applyCommonFactors(a);

  // Intersect factor lists (with multiplicity) to find common factors.
  // Since we are intersecting, it's enough to start with the first addend and
  // intersect with the rest.
  SmallVector<AffineExpr> common;
  flattenBinary(addends[0], AffineExprKind::Mul, common);
  for (int64_t i = 1, e = addends.size(); i < e; i++) {
    // Flatten the addend into a list of factors.
    SmallVector<AffineExpr> factors;
    flattenBinary(addends[i], AffineExprKind::Mul, factors);

    // Intersect the common factors with the factors of the addend.
    for (AffineExpr &term : common) {
      if (!term)
        continue;
      auto it = llvm::find(factors, term);
      // If the factor is not found, set it to nullptr to remove it from the
      // common factors.
      if (it == factors.end()) {
        term = nullptr;
        continue;
      }
      // Remove the factor from the list of factors to avoid duplicate
      // intersection.
      *it = nullptr;
    }
  }

  // Drop trivial factors.
  llvm::erase_if(common, [](AffineExpr e) {
    if (!e)
      return true;
    auto cst = dyn_cast<AffineConstantExpr>(e);
    return cst && cst.getValue() == 1;
  });

  // If there are no common factors, return the sum of the addends.
  if (common.empty())
    return computeSum(expr.getContext(), addends);

  // Divide each addend by the common factors, then recurse for nested factors.
  SmallVector<AffineExpr> divided(addends);
  for (AffineExpr cf : common) {
    for (AffineExpr &d : divided)
      d = divideByFactor(d, cf);
  }

  // Recursively apply common factors to the divided addends.
  for (AffineExpr &d : divided)
    d = applyCommonFactors(d);

  return computeProduct(expr.getContext(), common) *
         computeSum(expr.getContext(), divided);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

AffineExpr mlir::aster::factorizeAffineExpr(AffineExpr expr) {
  // There's nothing to do for non-binary expressions.
  auto binOp = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binOp)
    return expr;

  MLIRContext *ctx = expr.getContext();
  AffineExprKind kind = binOp.getKind();

  // For Mod/FloorDiv/CeilDiv: recurse into sub-expressions and rebuild.
  if (kind != AffineExprKind::Add && kind != AffineExprKind::Mul) {
    AffineExpr lhs = factorizeAffineExpr(binOp.getLHS());
    AffineExpr rhs = factorizeAffineExpr(binOp.getRHS());
    switch (kind) {
    case AffineExprKind::Mod:
      return lhs % rhs;
    case AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      llvm_unreachable("unexpected AffineExprKind");
    }
  }

  // For Add/Mul: expand to polynomial, greedy-factor, then apply common-factor
  // post-processing to fixpoint.
  Poly poly;
  affineToPoly(expr, poly);
  AffineExpr result = greedyFactor(std::move(poly), ctx);
  AffineExpr prev;
  do {
    prev = result;
    result = applyCommonFactors(result);
  } while (result != prev);
  return result;
}

//===----------------------------------------------------------------------===//
// FactorizeAffineExpr pass
//===----------------------------------------------------------------------===//

void FactorizeAffineExpr::runOnOperation() {
  IRRewriter rewriter(&getContext());
  getOperation()->walk([&](affine::AffineApplyOp op) {
    AffineMap map = op.getAffineMap();
    assert(map.getNumResults() == 1 && "affine.apply must have one result");
    AffineExpr original = map.getResult(0);
    AffineExpr factorized = aster::factorizeAffineExpr(original);
    if (factorized == original)
      return;
    AffineMap newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                      factorized, &getContext());
    rewriter.setInsertionPoint(op);
    auto newOp = affine::AffineApplyOp::create(rewriter, op.getLoc(), newMap,
                                               op.getOperands());
    rewriter.replaceOp(op, newOp.getResult());
  });
}
