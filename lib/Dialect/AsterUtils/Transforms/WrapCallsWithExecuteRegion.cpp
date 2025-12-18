//===- WrapCallsWithExecuteRegion.cpp - Wrap calls with execute_region ---===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "aster/Dialect/AsterUtils/Transforms/Transforms.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_WRAPCALLSWITHEXECUTEREGION
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {
//===----------------------------------------------------------------------===//
// WrapCallsWithExecuteRegion pass
//===----------------------------------------------------------------------===//
struct WrapCallsWithExecuteRegion
    : public aster_utils::impl::WrapCallsWithExecuteRegionBase<
          WrapCallsWithExecuteRegion> {
public:
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void WrapCallsWithExecuteRegion::runOnOperation() {
  wrapCallsWithExecuteRegion(getOperation());
}
