//===- LayoutOps.cpp - Layout operations implementation -----------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/IR/LayoutOps.h"
#include "aster/Dialect/Layout/IR/LayoutAttrs.h"
#include "aster/Dialect/Layout/IR/LayoutDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::layout;

//===----------------------------------------------------------------------===//
// Dialect definition (constructor, TypeID)
//===----------------------------------------------------------------------===//

#include "aster/Dialect/Layout/IR/LayoutDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void LayoutDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "aster/Dialect/Layout/IR/LayoutOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aster/Dialect/Layout/IR/LayoutOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

LogicalResult ApplyOp::verify() {
  size_t numCoords = getCoords().size();
  int64_t flatRank = getLayout().getFlatRank();
  bool ok = numCoords == 1 || numCoords == static_cast<size_t>(flatRank);
  if (!ok)
    return emitOpError() << "expected 1 (linear) or " << flatRank
                         << " (decomposed) coords, got " << numCoords;
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadValueOffsetsOp
//===----------------------------------------------------------------------===//

LogicalResult ThreadValueOffsetsOp::verify() {
  int64_t valSize = getValueLayout().getSize();
  if (static_cast<int64_t>(getResults().size()) != valSize)
    return emitOpError() << "expected " << valSize
                         << " results (= value_layout.size()), got "
                         << getResults().size();

  // Injectivity 1. flatten both layouts, rejecting broadcast modes (size > 1,
  // stride 0).
  SmallVector<std::pair<int64_t, int64_t>> modes;
  auto collect = [&](LayoutAttr lay, StringRef name) -> LogicalResult {
    SmallVector<int64_t> shape, stride;
    lay.flatten(shape, stride);
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] > 1 && stride[i] == 0)
        return emitOpError() << name
                             << " has a zero stride on a mode of "
                                "size "
                             << shape[i] << "; broadcasts and aliases";
      if (shape[i] > 1)
        modes.emplace_back(shape[i], stride[i]);
    }
    return success();
  };
  if (failed(collect(getThreadLayout(), "thread_layout")))
    return failure();
  if (failed(collect(getValueLayout(), "value_layout")))
    return failure();

  // Injectivity 2. drop size-1 modes, sort by stride, then sweep cumulative
  // max-reachable: a sorted strided layout with positive strides is injective
  // iff:
  //       sum_{j<=i} (size_j - 1) * stride_j < stride_{i+1}.
  llvm::sort(modes, [](auto a, auto b) { return a.second < b.second; });
  int64_t cumulativeMax = 0;
  for (size_t i = 0; i + 1 < modes.size(); ++i) {
    cumulativeMax += (modes[i].first - 1) * modes[i].second;
    int64_t nextStride = modes[i + 1].second;
    if (cumulativeMax >= nextStride)
      return emitOpError() << "thread_layout o value_layout is not injective: "
                           << "cumulative max offset " << cumulativeMax << " "
                           << "through modes with stride <= " << modes[i].second
                           << " overlaps the next mode at stride "
                           << nextStride;
  }
  return success();
}
