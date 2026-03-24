//===- Legalizer.cpp - Legalize operations for code generation ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "aster/CodeGen/Passes.h"

#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
#define GEN_PASS_DEF_LEGALIZER
#include "aster/CodeGen/Passes.h.inc"
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// Legalizer pass
//===----------------------------------------------------------------------===//
struct Legalizer : public aster::impl::LegalizerBase<Legalizer> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Legalizer pass
//===----------------------------------------------------------------------===//

static LogicalResult
runGreedyRewriter(Operation *op,
                  std::function<void(RewritePatternSet &)> populatePatterns,
                  bool topDownTraversal = true) {
  RewritePatternSet patterns(op->getContext());
  populatePatterns(patterns);
  auto config = GreedyRewriteConfig()
                    .enableFolding()
                    .enableConstantCSE()
                    .setUseTopDownTraversal(topDownTraversal);
  if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
    return failure();
  return success();
}

void Legalizer::runOnOperation() {
  Operation *op = getOperation();

  // Legalize operations that don't require type conversion.
  if (failed(runGreedyRewriter(op, [&](RewritePatternSet &patterns) {
        amdgcn::populateAMDGPULegalizationPatterns(patterns);
      })))
    return signalPassFailure();

  // Legalize operations that require type conversion.
  {
    LegalizerTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    RewritePatternSet conversionPatterns(&getContext());
    ConversionConfig config;
    config.allowPatternRollback = false;
    amdgcn::populateAMDGPUTypeLegalizationPatterns(converter, target,
                                                   conversionPatterns);
    if (failed(applyPartialConversion(
            getOperation(), target,
            FrozenRewritePatternSet(std::move(conversionPatterns)), config)))
      return signalPassFailure();
  }

  // Fold ptr.to_ptr(materialization_cast(x : ptr -> memref)) to identity.
  // After type conversion, extract_strided_metadata produces a base memref
  // from a materialized cast of the already-converted ptr arg. ptr.to_ptr
  // then converts it right back. With matching address spaces this is a no-op.
  //
  // Note: this is currently used when lowering from memref in the mlir-air path
  // and would ideally be more composable / not use UnrealizedConversionCastOp.
  // However, removing UnrealizedConversionCastOp here runs into a deeper rabbit
  // hole where we need to remove UnrealizedConversionCastOp in many other
  // places in aster that are currently load-bearing.
  {
    SmallVector<Operation *> toErase;
    getOperation()->walk([&](ptr::ToPtrOp toPtrOp) {
      auto cast = toPtrOp.getPtr().getDefiningOp<UnrealizedConversionCastOp>();
      if (!cast || cast.getNumOperands() != 1 || cast.getNumResults() != 1)
        return;
      if (!isa<MemRefType>(cast.getResultTypes()[0]))
        return;
      Value src = cast.getOperand(0);
      if (src.getType() != toPtrOp.getType())
        return;
      toPtrOp.replaceAllUsesWith(src);
      toErase.push_back(toPtrOp);
    });
    for (auto *op : toErase)
      op->erase();
    SmallVector<Operation *> deadOps;
    getOperation()->walk([&](UnrealizedConversionCastOp op) {
      if (op.use_empty())
        deadOps.push_back(op);
    });
    for (auto *op : deadOps)
      op->erase();
  }

  // Attach AMDGCN pointer size data layout to the module.
  // LDS (local) pointers are 32-bit, global pointers are 64-bit.
  if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
    MLIRContext *ctx = &getContext();
    auto rwAccess = amdgcn::AccessKind::ReadWrite;

    auto localEntry = DataLayoutEntryAttr::get(
        ptr::PtrType::get(ctx,
                          amdgcn::AddressSpaceAttr::get(
                              ctx, amdgcn::AddressSpaceKind::Local, rwAccess)),
        ptr::SpecAttr::get(ctx, /*size=*/32, /*abi=*/32, /*preferred=*/32));

    auto globalEntry = DataLayoutEntryAttr::get(
        ptr::PtrType::get(ctx,
                          amdgcn::AddressSpaceAttr::get(
                              ctx, amdgcn::AddressSpaceKind::Global, rwAccess)),
        ptr::SpecAttr::get(ctx, /*size=*/64, /*abi=*/64, /*preferred=*/64));

    moduleOp->setAttr(DLTIDialect::kDataLayoutAttrName,
                      DataLayoutSpecAttr::get(ctx, {localEntry, globalEntry}));
  }
}
