//===- CodeGenPass.cpp ----------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Passes.h"

#include "aster/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/LSIR/CodeGen/CodeGen.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Transforms/Utils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::aster {
#define GEN_PASS_DEF_CODEGEN
#include "aster/CodeGen/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// CodeGen pass
//===----------------------------------------------------------------------===//
struct CodeGen : public aster::impl::CodeGenBase<CodeGen> {
public:
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// CodeGen pass
//===----------------------------------------------------------------------===//

void CodeGen::getDependentDialects(DialectRegistry &registry) const {
  // TODO: Hide these functions behind an interface ala ConvertToLLVM.
  amdgcn::getDependentCodeGenDialects(registry);
  lsir::getDependentCodeGenDialects(registry);
}

void CodeGen::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  FailureOr<ConvertCodeGenState> state = ConvertCodeGenState::create(op);
  if (failed(state))
    return signalPassFailure();
  CodeGenConverter converter(*state);
  ConversionTarget target(getContext());
  // TODO: Hide these functions behind an interface ala ConvertToLLVM.
  amdgcn::populateCodeGenPatterns(converter, patterns, target);
  lsir::populateCodeGenPatterns(converter, patterns, target);
  populateFuncConversionPatterns(converter, target, patterns);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
  SmallVector<UnrealizedConversionCastOp> ops;
  getOperation()->walk(
      [&](UnrealizedConversionCastOp castOp) { ops.push_back(castOp); });
  reconcileUnrealizedCasts(ops);
}
