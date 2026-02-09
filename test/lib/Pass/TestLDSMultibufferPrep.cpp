//===- TestLDSMultibufferPrep.cpp - Test LDS Multi-Buffer Prep -----------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass wrapper for the LDS multi-buffer prep transformation.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Transforms/Transforms.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster::test {
#define GEN_PASS_DEF_TESTLDSMULTIBUFFERPREP
#include "Passes.h.inc"
} // namespace mlir::aster::test

namespace {
struct TestLDSMultibufferPrep
    : public mlir::aster::test::impl::TestLDSMultibufferPrepBase<
          TestLDSMultibufferPrep> {
  using TestLDSMultibufferPrepBase::TestLDSMultibufferPrepBase;

  void runOnOperation() override {
    if (failed(mlir::aster::prepareLDSMultibuffers(getOperation())))
      return signalPassFailure();
  }
};
} // namespace
