// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PromoteAllocsToFuncArgs.cpp - alloc → function argument promotion --===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static void promoteAllocsInFunc(func::FuncOp funcOp) {
  if (!funcOp->hasAttr("gpu.kernel"))
    return;

  SmallVector<memref::AllocOp> allocsToPromote;
  // Walk ALL allocs in the function, including those nested in scf.if
  // or other control flow (e.g., from air-split-launch-for-padding).
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (allocOp.getMemref().getType().getMemorySpace())
      return;
    if (!allocOp.getMemref().getType().hasStaticShape())
      return;
    allocsToPromote.push_back(allocOp);
  });

  if (allocsToPromote.empty())
    return;

  // Deduplicate allocs with the same type: share a single promoted arg.
  // Multiple allocs of the same type (e.g., from split-launch scf.if
  // branches) can safely share one buffer since only one branch executes.
  DenseMap<Type, unsigned> typeToArgIdx;
  SmallVector<memref::AllocOp> uniqueAllocs;
  SmallVector<unsigned> allocArgMapping; // maps each alloc → arg index

  auto funcTy = funcOp.getFunctionType();
  SmallVector<Type> newArgTypes(funcTy.getInputs());
  for (auto allocOp : allocsToPromote) {
    auto ty = allocOp.getMemref().getType();
    auto it = typeToArgIdx.find(ty);
    if (it == typeToArgIdx.end()) {
      unsigned idx = newArgTypes.size();
      typeToArgIdx[ty] = idx;
      newArgTypes.push_back(ty);
      uniqueAllocs.push_back(allocOp);
      allocArgMapping.push_back(idx);
    } else {
      allocArgMapping.push_back(it->second);
    }
  }

  funcOp.setFunctionType(
      FunctionType::get(funcOp.getContext(), newArgTypes, funcTy.getResults()));

  auto &entryBlock = funcOp.getBody().front();
  // Add one block arg per unique type.
  SmallVector<Value> newArgs;
  for (auto allocOp : uniqueAllocs) {
    auto newArg =
        entryBlock.addArgument(allocOp.getMemref().getType(), allocOp.getLoc());
    newArgs.push_back(newArg);
  }
  // Replace each alloc with the corresponding shared arg.
  for (unsigned i = 0; i < allocsToPromote.size(); ++i) {
    unsigned argIdx = allocArgMapping[i] - funcTy.getNumInputs();
    allocsToPromote[i].getResult().replaceAllUsesWith(newArgs[argIdx]);
  }

  // Erase the allocs and their initialization ops (fill, copy into padded
  // buffer). The host is responsible for pre-initializing the workspace.
  for (auto allocOp : llvm::reverse(allocsToPromote)) {
    for (auto user :
         llvm::make_early_inc_range(allocOp.getResult().getUsers())) {
      if (isa<memref::DeallocOp>(user))
        user->erase();
    }
    allocOp->erase();
  }

  // Erase linalg.map (zero-fill) and memref.copy (padding init copy)
  // on the promoted workspace args. The host pre-initializes them.
  // Walk ALL ops (including nested in scf.if) to find operations on
  // promoted workspace args.
  DenseSet<Value> promotedArgs(newArgs.begin(), newArgs.end());
  SmallVector<Operation *> toErase;
  funcOp.walk([&](Operation *op) {
    if (auto mapOp = dyn_cast<linalg::MapOp>(op)) {
      for (Value out : mapOp.getDpsInits())
        if (promotedArgs.contains(out))
          toErase.push_back(op);
    } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      // Only erase copies INTO workspace (dst is workspace).
      // Keep copies FROM workspace to global (the result copy-back).
      Value dst = copyOp.getTarget();
      auto dstSv = dst.getDefiningOp<memref::SubViewOp>();
      bool dstIsWs = promotedArgs.contains(dst) ||
                     (dstSv && promotedArgs.contains(dstSv.getSource()));
      if (dstIsWs)
        toErase.push_back(op);
    }
  });
  for (auto *op : llvm::reverse(toErase))
    op->erase();
}

struct PromoteAllocsToFuncArgs
    : public PassWrapper<PromoteAllocsToFuncArgs, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteAllocsToFuncArgs)
  StringRef getArgument() const override {
    return "promote-allocs-to-func-args";
  }
  StringRef getDescription() const override {
    return "Promote function-level memref.alloc to function arguments";
  }
  void runOnOperation() override {
    getOperation()->walk([](func::FuncOp f) { promoteAllocsInFunc(f); });
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createPromoteAllocsToFuncArgs() {
  return std::make_unique<PromoteAllocsToFuncArgs>();
}
} // namespace mlir::aster::mlir_air
