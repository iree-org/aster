//===- InlineExecuteRegion.cpp - Inline execute_region operations --------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/Transforms/Transforms.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_INLINEEXECUTEREGION
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// InlineExecuteRegion pass
//===----------------------------------------------------------------------===//
struct InlineExecuteRegion
    : public aster_utils::impl::InlineExecuteRegionBase<InlineExecuteRegion> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void InlineExecuteRegion::runOnOperation() {
  inlineExecuteRegions(getOperation());
}
