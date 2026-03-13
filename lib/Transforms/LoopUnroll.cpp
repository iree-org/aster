//===- LoopUnroll.cpp -----------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::aster {
#define GEN_PASS_DEF_LOOPUNROLL
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
constexpr const char *kUnrollFactorAttr = "unroll_factor";

/// Extract unroll factors from an attribute. Returns factors if the attribute
/// is a valid unroll_factor (IntegerAttr or dense int array), otherwise
/// returns std::nullopt.
std::optional<SmallVector<int64_t>> getUnrollFactorsFromAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;

  // Single integer attribute
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    int64_t factor = intAttr.getValue().getSExtValue();
    if (factor <= 1)
      return std::nullopt;
    return SmallVector<int64_t>{factor};
  }

  // Dense int array (array<i64: ...>)
  if (auto arrayAttr = dyn_cast<DenseI64ArrayAttr>(attr)) {
    SmallVector<int64_t> factors(arrayAttr.asArrayRef().begin(),
                                 arrayAttr.asArrayRef().end());
    factors.erase(llvm::remove_if(factors, [](int64_t f) { return f <= 1; }),
                  factors.end());
    if (factors.empty())
      return std::nullopt;
    return factors;
  }

  return std::nullopt;
}

/// Normalize factors: sort descending, remove duplicates and factors <= 1.
void normalizeFactors(SmallVector<int64_t> &factors) {
  llvm::sort(factors, std::greater<int64_t>());
  factors.erase(
      llvm::remove_if(factors, [](int64_t factor) { return factor <= 1; }),
      factors.end());
  factors.erase(llvm::unique(factors), factors.end());
}

//===----------------------------------------------------------------------===//
// LoopUnroll pass
//===----------------------------------------------------------------------===//
struct LoopUnroll : public mlir::aster::impl::LoopUnrollBase<LoopUnroll> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// LoopUnroll pass
//===----------------------------------------------------------------------===//

void LoopUnroll::runOnOperation() {
  Operation *op = getOperation();

  // Default factors from pass option.
  SmallVector<int64_t> defaultFactors(unrollFactors.begin(),
                                      unrollFactors.end());
  normalizeFactors(defaultFactors);

  WalkResult walkResult = op->walk([&](scf::ForOp forOp) {
    std::optional<APInt> maybeTripCount = forOp.getStaticTripCount();
    if (!maybeTripCount.has_value())
      return WalkResult::advance();

    int64_t tripCount = maybeTripCount->getSExtValue();

    // Use loop's unroll_factor attribute first if present.
    SmallVector<int64_t> factors;
    if (Attribute attr = forOp->getAttr(kUnrollFactorAttr)) {
      if (auto attrFactors = getUnrollFactorsFromAttr(attr)) {
        factors = std::move(*attrFactors);
        normalizeFactors(factors);
      }
    }
    if (factors.empty())
      factors = defaultFactors;

    for (int64_t factor : factors) {
      if (tripCount % factor != 0)
        continue;

      if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(factor)))) {
        op->emitError() << "failed to unroll loop " << forOp.getLoc()
                        << " by factor " << factor;
        return WalkResult::interrupt();
      }
      break;
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}
