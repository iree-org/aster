//===- IntRangeImpl.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Interfaces/GPUFuncInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

static std::pair<ArrayRef<int32_t>, ArrayRef<int32_t>>
getLaunchBounds(Operation *op) {
  auto gpuFunc = op->getParentOfType<GPUFuncInterface>();
  if (!gpuFunc)
    return {};
  return {gpuFunc.getGridDims(), gpuFunc.getBlockDims()};
}

void BlockDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (blockDims.empty()) {
    return setResultRange(getResult(),
                          ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                                   llvm::APInt(32, 1024, false),
                                                   false));
  }
  Dim dim = getDim();
  int32_t size = blockDims.size() > static_cast<size_t>(dim)
                     ? blockDims[static_cast<int32_t>(dim)]
                     : 1;
  setResultRange(getResult(),
                 ConstantIntRanges::constant(llvm::APInt(32, size, false)));
}

void BlockIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (gridDims.empty()) {
    return setResultRange(getResult(),
                          ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                                   llvm::APInt(32, 1024, false),
                                                   false));
  }
  Dim dim = getDim();
  int32_t size = gridDims.size() > static_cast<size_t>(dim)
                     ? gridDims[static_cast<int32_t>(dim)]
                     : 1;
  // block_id ranges [0, gridDims[dim] - 1] (0-indexed).
  setResultRange(getResult(), ConstantIntRanges::range(
                                  llvm::APInt(32, 0, false),
                                  llvm::APInt(32, size - 1, false), false));
}

void GridDimOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                  SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (gridDims.empty()) {
    return setResultRange(
        getResult(),
        ConstantIntRanges::range(
            llvm::APInt(32, 0, false),
            llvm::APInt(32, std::numeric_limits<int32_t>::max(), false),
            false));
  }
  Dim dim = getDim();
  int32_t size = gridDims.size() > static_cast<size_t>(dim)
                     ? gridDims[static_cast<int32_t>(dim)]
                     : 1;
  setResultRange(getResult(),
                 ConstantIntRanges::constant(llvm::APInt(32, size, false)));
}

void ThreadIdOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  auto [gridDims, blockDims] = getLaunchBounds(getOperation());
  if (blockDims.empty()) {
    return setResultRange(getResult(),
                          ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                                   llvm::APInt(32, 1024, false),
                                                   false));
  }
  Dim dim = getDim();
  int32_t size = blockDims.size() > static_cast<size_t>(dim)
                     ? blockDims[static_cast<int32_t>(dim)]
                     : 1;
  // thread_id ranges [0, blockDims[dim] - 1] (0-indexed).
  setResultRange(getResult(), ConstantIntRanges::range(
                                  llvm::APInt(32, 0, false),
                                  llvm::APInt(32, size - 1, false), false));
}

/// Infer the result ranges of the assume_range operation.
void AssumeRangeOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges,
                                      SetIntRangeFn setResultRange) {
  // Get min/max bounds - for dynamic bounds, we need to use the input range
  // from dataflow analysis. For static bounds, use the attribute value.

  unsigned width = IndexType::kInternalStorageBitWidth;
  if (getResult().getType().isSignlessInteger())
    width = getResult().getType().getIntOrFloatBitWidth();

  llvm::APInt umin = llvm::APInt::getMinValue(width);
  llvm::APInt umax = llvm::APInt::getMaxValue(width);
  llvm::APInt smin = llvm::APInt::getSignedMinValue(width);
  llvm::APInt smax = llvm::APInt::getSignedMaxValue(width);

  // Get the min range.
  if (hasStaticMin()) {
    umin = getStaticMinAttr().getValue();
    smin = getStaticMinAttr().getValue();
  } else if (getDynamicMin()) {
    umin = ranges[1].umin();
    smin = ranges[1].smin();
  }

  // Get the max range.
  if (hasStaticMax()) {
    umax = getStaticMaxAttr().getValue();
    smax = getStaticMaxAttr().getValue();
  } else if (getDynamicMax()) {
    size_t maxIdx = getDynamicMin() ? 2 : 1;
    if (ranges.size() > maxIdx) {
      umax = ranges[maxIdx].umax();
      smax = ranges[maxIdx].smax();
    }
  }

  // If the unsigned range is invalid, set it to max range.
  if (umin.ugt(umax)) {
    LDBG() << " - Invalid unsigned range, setting to max range";
    umin = llvm::APInt::getMinValue(width);
    umax = llvm::APInt::getMaxValue(width);
  }

  // If the signed range is invalid, set it to max range.
  if (smin.sgt(smax)) {
    LDBG() << " - Invalid signed range, setting to max range";
    smin = llvm::APInt::getSignedMinValue(width);
    smax = llvm::APInt::getSignedMaxValue(width);
  }

  setResultRange(getResult(), ConstantIntRanges(umin, umax, smin, smax));
}

void AssumeRangeOp::populateBoundsForIndexValue(
    Value value, ValueBoundsConstraintSet &cstr) {
  assert(value == getResult() &&
         "inferring for value that isn't the assume_range op's result");
  if (!value.getType().isIndex())
    return;
  if (OpFoldResult min = getMin())
    cstr.bound(value) >= min;
  if (OpFoldResult max = getMax())
    cstr.bound(value) <= max;
}

void AssumeUniformOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> ranges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), ranges.front());
}
