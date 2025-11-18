//===- ConstexprExpansion.cpp ---------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_CONSTEXPREXPANSION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ConstexprExpansion pass
//===----------------------------------------------------------------------===//
struct ConstexprExpansion
    : public mlir::aster::amdgcn::impl::ConstexprExpansionBase<
          ConstexprExpansion> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConstexprExpansion pass
//===----------------------------------------------------------------------===//

void ConstexprExpansion::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("amdgcn.constexpr"))
      return;
    if (failed(loopUnrollFull(forOp))) {
      forOp.emitWarning()
          << "failed to unroll scf.for marked with amdgcn.constexpr attribute";
    }
  });
}
