//===- ReplaceConstantGPUDims.cpp - Replace GPU dims with constants ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"

#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::aster {
#define GEN_PASS_DEF_REPLACECONSTANTGPUDIMS
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ReplaceConstantGPUDims pass
//===----------------------------------------------------------------------===//
struct ReplaceConstantGPUDims
    : public aster::impl::ReplaceConstantGPUDimsBase<ReplaceConstantGPUDims> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Replace dimension ops within a GPU function with constants.
  void expandDimsInFunction(GPUFuncInterface gpuFunc);
};
} // namespace

/// Get the dimension value from the dims array based on the GPU dimension.
static int32_t getDimValue(ArrayRef<int32_t> dims, gpu::Dimension dim) {
  if (dims.empty())
    return 0;
  switch (dim) {
  case gpu::Dimension::x:
    return dims.size() > 0 ? dims[0] : 1;
  case gpu::Dimension::y:
    return dims.size() > 1 ? dims[1] : 1;
  case gpu::Dimension::z:
    return dims.size() > 2 ? dims[2] : 1;
  }
  llvm_unreachable("unknown dimension");
}

void ReplaceConstantGPUDims::expandDimsInFunction(GPUFuncInterface gpuFunc) {
  ArrayRef<int32_t> blockDims = gpuFunc.getBlockDims();
  ArrayRef<int32_t> gridDims = gpuFunc.getGridDims();

  // If neither block nor grid dims are set, nothing to do.
  if (blockDims.empty() && gridDims.empty())
    return;

  Operation *funcOp = gpuFunc.getOperation();
  IRRewriter rewriter(funcOp->getContext());

  // Collect ops to replace (avoid modifying while iterating).
  SmallVector<gpu::BlockDimOp> blockDimOps;
  SmallVector<gpu::GridDimOp> gridDimOps;

  funcOp->walk([&](Operation *op) {
    if (auto blockDimOp = dyn_cast<gpu::BlockDimOp>(op)) {
      if (!blockDims.empty())
        blockDimOps.push_back(blockDimOp);
    } else if (auto gridDimOp = dyn_cast<gpu::GridDimOp>(op)) {
      if (!gridDims.empty())
        gridDimOps.push_back(gridDimOp);
    }
  });

  // Replace block dim ops with constants.
  for (gpu::BlockDimOp op : blockDimOps) {
    int32_t value = getDimValue(blockDims, op.getDimension());
    rewriter.setInsertionPoint(op);
    Value constVal =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), value);
    rewriter.replaceOp(op, constVal);
  }

  // Replace grid dim ops with constants.
  for (gpu::GridDimOp op : gridDimOps) {
    int32_t value = getDimValue(gridDims, op.getDimension());
    rewriter.setInsertionPoint(op);
    Value constVal =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), value);
    rewriter.replaceOp(op, constVal);
  }
}

void ReplaceConstantGPUDims::runOnOperation() {
  Operation *moduleOp = getOperation();
  moduleOp->walk([&](Operation *op) {
    if (auto gpuFunc = dyn_cast<GPUFuncInterface>(op))
      expandDimsInFunction(gpuFunc);
  });
}
