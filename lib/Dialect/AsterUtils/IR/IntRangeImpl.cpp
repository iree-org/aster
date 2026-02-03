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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/APInt.h"
#include <limits>

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
  setResultRange(getResult(),
                 ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                          llvm::APInt(32, size, false), false));
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
  setResultRange(getResult(),
                 ConstantIntRanges::range(llvm::APInt(32, 0, false),
                                          llvm::APInt(32, size, false), false));
}

void AssumeRangeOp::inferResultRanges(ArrayRef<ConstantIntRanges> ranges,
                                      SetIntRangeFn setResultRange) {
  // Get min/max bounds - for dynamic bounds, we need to use the input range
  // from dataflow analysis. For static bounds, use the attribute value.
  std::optional<APInt> min;
  std::optional<APInt> max;

  // Check for static min bound
  if (hasStaticMin())
    min = getStaticMinAttr().getValue();
  else if (getDynamicMin() && ranges.size() > 1)
    min = ranges[1].smin(); // Use the min from dataflow analysis

  // Check for static max bound (index into ranges depends on whether
  // dynamic_min exists)
  if (hasStaticMax()) {
    max = getStaticMaxAttr().getValue();
  } else if (getDynamicMax()) {
    size_t maxIdx = getDynamicMin() ? 2 : 1;
    if (ranges.size() > maxIdx)
      max = ranges[maxIdx].smax(); // Use the max from dataflow analysis
  }

  if (!min && !max) {
    setResultRange(getResult(), ranges.front());
    return;
  }
  unsigned width = getResult().getType().getIntOrFloatBitWidth();
  ConstantIntRanges inRange = ranges.front();
  if (max)
    max = max->sextOrTrunc(width);
  if (min)
    min = min->sextOrTrunc(width);
  if (!min) {
    setResultRange(getResult(),
                   inRange.intersection(ConstantIntRanges::fromSigned(
                       APInt::getSignedMinValue(width), *max)));
    return;
  }
  if (!max) {
    setResultRange(getResult(),
                   inRange.intersection(ConstantIntRanges::fromSigned(
                       *min, APInt::getSignedMaxValue(width))));
    return;
  }
  setResultRange(getResult(), inRange.intersection(
                                  ConstantIntRanges::fromSigned(*min, *max)));
}

void AssumeUniformOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> ranges, SetIntLatticeFn setResultRange) {
  setResultRange(getResult(), ranges.front());
}
