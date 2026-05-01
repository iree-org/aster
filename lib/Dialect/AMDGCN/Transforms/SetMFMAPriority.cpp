//===- SetMFMAPriority.cpp - Insert s_setprio around MFMA groups ----------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/IR/Builders.h"

namespace mlir::aster::amdgcn {
#define GEN_PASS_DEF_SETMFMAPRIORITY
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace mlir::aster::amdgcn

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

static bool isMFMA(Operation *op) {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return false;
  return instOp.hasAnyProps({InstProp::Mma, InstProp::ScaledMma});
}

static bool isMemoryOp(Operation *op) {
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp || instOp.getOpCode() == OpCode::Invalid)
    return false;
  return instOp.hasAnyProps(
      {InstProp::IsVmem, InstProp::Dsmem, InstProp::Smem});
}

static bool blockHasSetprio(Block &block) {
  for (auto &op : block) {
    if (isa<SSetprio>(&op))
      return true;
  }
  return false;
}

struct MFMAGroup {
  Operation *firstMfma;
  Operation *lastMfma;
};

static SmallVector<MFMAGroup> findMFMAGroups(Block &block) {
  SmallVector<MFMAGroup> groups;
  Operation *firstMfma = nullptr;
  Operation *lastMfma = nullptr;

  for (auto &op : block) {
    if (isMFMA(&op)) {
      if (!firstMfma)
        firstMfma = &op;
      lastMfma = &op;
    } else if (isMemoryOp(&op)) {
      if (firstMfma)
        groups.push_back({firstMfma, lastMfma});
      firstMfma = nullptr;
      lastMfma = nullptr;
    }
  }
  // Close trailing group at block end.
  if (firstMfma)
    groups.push_back({firstMfma, lastMfma});

  return groups;
}

struct SetMFMAPriority
    : public amdgcn::impl::SetMFMAPriorityBase<SetMFMAPriority> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void SetMFMAPriority::runOnOperation() {
  Operation *op = getOperation();
  IRRewriter rewriter(op->getContext());

  op->walk([&](Block *blockPtr) {
    Block &block = *blockPtr;
    {
      if (blockHasSetprio(block))
        return;

      auto groups = findMFMAGroups(block);
      for (auto &group : groups) {
        rewriter.setInsertionPoint(group.firstMfma);
        SSetprio::create(rewriter, group.firstMfma->getLoc(),
                         static_cast<uint16_t>(1));
        rewriter.setInsertionPointAfter(group.lastMfma);
        SSetprio::create(rewriter, group.lastMfma->getLoc(),
                         static_cast<uint16_t>(0));
      }
    }
  });
}
