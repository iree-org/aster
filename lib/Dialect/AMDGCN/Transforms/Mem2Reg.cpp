//===- Mem2Reg.cpp --------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Mem2Reg.h"

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
//===----------------------------------------------------------------------===//
// Mem2Reg pass
//===----------------------------------------------------------------------===//
struct Mem2Reg : public amdgcn::impl::Mem2RegBase<Mem2Reg> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Mem2Reg pass
//===----------------------------------------------------------------------===//

/// Runs the upstream Mem2Reg transformation on the given operation.
/// This code was adapted from: llvm-project/mlir/lib/Transforms/Mem2Reg.cpp
static bool runUpstreamMem2Reg(RewriterBase &rewriter, Operation *op,
                               const DataLayout &dataLayout,
                               DominanceInfo &dominance) {
  bool changed = false;
  for (Region &region : op->getRegions()) {
    if (region.getBlocks().empty())
      continue;
    SmallVector<PromotableAllocationOpInterface> allocators;
    region.walk([&](PromotableAllocationOpInterface allocator) {
      // Skip over any allocator that does not allocate a register type, and
      // it's not a `memref.alloca`.
      if (auto aOp = dyn_cast<memref::AllocaOp>(allocator.getOperation());
          !aOp || !isa<RegisterTypeInterface>(aOp.getType().getElementType()))
        return;
      allocators.emplace_back(allocator);
    });
    rewriter.setInsertionPointToStart(&region.front());
    if (succeeded(tryToPromoteMemorySlots(allocators, rewriter, dataLayout,
                                          dominance)))
      changed = true;
  }
  return changed;
}

static RegisterTypeInterface getRegisterType(RegisterTypeInterface base,
                                             Register reg) {
  return base.cloneRegisterType(reg);
}

void Mem2Reg::runOnOperation() {
  Operation *op = getOperation();
  auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(op);
  auto &dominance = getAnalysis<DominanceInfo>();
  IRRewriter rewriter(&getContext());
  if (!runUpstreamMem2Reg(rewriter, op, dataLayout, dominance))
    markAllAnalysesPreserved();
  // `mem2reg` Might allocate `ub.poison` ops to represent uninitialized
  // register values. Replace them with `amdgcn.alloca` ops.
  op->walk([&rewriter](ub::PoisonOp pOp) {
    if (!isa<RegisterTypeInterface>(pOp.getType()))
      return;
    rewriter.setInsertionPoint(pOp);
    auto regType = dyn_cast<RegisterTypeInterface>(pOp.getType());
    if (regType.isRegisterRange()) {
      rewriter.setInsertionPoint(pOp);
      SmallVector<Value> allocas;
      RegisterRange range = regType.getAsRange();
      for (int16_t i = 0; i < range.size(); ++i) {
        Register reg = regType.isRelocatable()
                           ? Register()
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
}
