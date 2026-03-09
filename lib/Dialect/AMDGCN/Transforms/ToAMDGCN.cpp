//===- ToAMDGCN.cpp -------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace aster {
namespace amdgcn {
#define GEN_PASS_DEF_TOAMDGCN
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//

struct ToAMDGCN : public amdgcn::impl::ToAMDGCNBase<ToAMDGCN> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//

static OwningOpRef<mlir::ModuleOp> getPDLModule(StringRef inputFile,
                                                MLIRContext *ctx) {
  if (inputFile.empty())
    return nullptr;
  return parseSourceFile<mlir::ModuleOp>(inputFile, ParserConfig(ctx));
}

void ToAMDGCN::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  OwningOpRef<mlir::ModuleOp> moduleRef =
      getPDLModule(inputFile, &getContext());
  if (moduleRef) {
    PDLPatternModule pdlPattern(std::move(moduleRef));
    patterns.add(std::move(pdlPattern));
  }
  populateToAMDGCNPatterns(patterns);
  if (failed(applyPatternsGreedily(op,
                                   FrozenRewritePatternSet(std::move(patterns)),
                                   GreedyRewriteConfig()
                                       .setUseTopDownTraversal()
                                       .enableConstantCSE()
                                       .enableFolding())))
    return signalPassFailure();

  // Set post-condition: no reg_cast ops remain.
  // Note: no_lsir_ops is NOT set here because some ToAMDGCN patterns
  // (ExtSI, ExtUI, PtrAdd) create new LSIR ops as intermediaries, and
  // in pipelined loop bodies these may not all get cleaned up within
  // the greedy rewrite.
  op->walk([&](amdgcn::ModuleOp moduleOp) {
    moduleOp.addNormalForms({NoRegCastOpsAttr::get(&getContext())});
  });
}
