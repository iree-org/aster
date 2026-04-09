//===- Mem2Reg.cpp --------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Post-processing pass for upstream --mem2reg: replaces ub.poison of ASTER
// register types with amdgcn.alloca (+ make_register_range for ranges).
// Upstream mem2reg handles the actual promotion; this pass only materializes
// uninitialized register values that upstream leaves as poison.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_MEM2REG
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct Mem2Reg : public amdgcn::impl::Mem2RegBase<Mem2Reg> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static RegisterTypeInterface getRegisterType(RegisterTypeInterface base,
                                             Register reg) {
  return base.cloneRegisterType(reg);
}

void Mem2Reg::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(&getContext());
  bool changed = false;
  // Replace ub.poison of register types with amdgcn.alloca.
  // Token and struct types stay as poison since they don't need allocation.
  op->walk([&rewriter, &changed](ub::PoisonOp pOp) {
    if (!isa<RegisterTypeInterface>(pOp.getType()))
      return;
    changed = true;
    rewriter.setInsertionPoint(pOp);
    auto regType = cast<RegisterTypeInterface>(pOp.getType());
    if (regType.isRegisterRange()) {
      SmallVector<Value> allocas;
      RegisterRange range = regType.getAsRange();
      for (int16_t i = 0; i < range.size(); ++i) {
        Register reg = !regType.hasAllocatedSemantics()
                           ? range.begin()
                           : Register(range.begin().getRegister() + i);
        allocas.push_back(amdgcn::AllocaOp::create(
            rewriter, pOp.getLoc(), getRegisterType(regType, reg)));
      }
      rewriter.replaceOpWithNewOp<amdgcn::MakeRegisterRangeOp>(
          pOp, pOp.getType(), allocas);
      return;
    }
    rewriter.replaceOpWithNewOp<amdgcn::AllocaOp>(pOp, pOp.getType());
  });
  if (!changed)
    markAllAnalysesPreserved();
}
