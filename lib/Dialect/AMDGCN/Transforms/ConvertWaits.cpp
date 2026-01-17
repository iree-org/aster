//===- ConvertWaits.cpp - Convert wait ops to hardware instructions ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNCONVERTWAITS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct AMDGCNConvertWaits
    : public mlir::aster::amdgcn::impl::AMDGCNConvertWaitsBase<
          AMDGCNConvertWaits> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void AMDGCNConvertWaits::runOnOperation() {
  // TODO: Implement conversion of wait operations to hardware instructions.
  // This pass should convert amdgcn.wait operations to appropriate hardware
  // wait instructions (s_waitcnt, s_wait_loadcnt, s_wait_storecnt, etc.)
  // based on the types of dependencies being waited on.
}
