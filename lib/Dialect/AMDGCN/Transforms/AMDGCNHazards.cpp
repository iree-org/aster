//===- AMDGCNHazards.cpp - AMDGCN hazards pass --------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/HazardAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/HazardManager.h"
#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster::amdgcn {
#define GEN_PASS_DEF_AMDGCNHAZARDS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace mlir::aster::amdgcn

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// AMDGCNHazards pass
//===----------------------------------------------------------------------===//
struct AMDGCNHazards : public amdgcn::impl::AMDGCNHazardsBase<AMDGCNHazards> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNHazards pass
//===----------------------------------------------------------------------===//

void AMDGCNHazards::runOnOperation() {
  Operation *op = getOperation();

  // Pre-condition: all registers must be allocated before NOP insertion.
  if (auto kernelOp = dyn_cast<KernelOp>(op)) {
    if (!kernelOp.hasNormalForm(
            AllRegistersAllocatedAttr::get(op->getContext()))) {
      op->emitError() << "amdgcn-hazards requires "
                         "#amdgcn.all_registers_allocated normal form";
      return signalPassFailure();
    }
  }

  // Get ISA version from the module, default to CDNA3 if not found.
  ISAVersion isaVersion = ISAVersion::CDNA3;
  if (auto moduleOp = op->getParentOfType<amdgcn::ModuleOp>())
    isaVersion = getIsaForTarget(moduleOp.getTarget());

  if (extraVNops > 128) {
    op->emitError() << "extraVNops must be between 0 and 128, got "
                    << extraVNops;
    return signalPassFailure();
  }

  if (extraSNops > 128) {
    op->emitError() << "extraSNops must be between 0 and 128, got "
                    << extraSNops;
    return signalPassFailure();
  }

  HazardManager hazardManager(op);
  SmallVector<HazardRaiserAttrInterface, 1> additionalHazardRaisers;
  if (extraVNops > 0 || extraSNops > 0) {
    additionalHazardRaisers.push_back(
        AllHazardAttr::get(op->getContext(), static_cast<int16_t>(extraVNops),
                           static_cast<int16_t>(extraSNops)));
  }
  hazardManager.populateHazardsFor(isaVersion, additionalHazardRaisers);

  // Run hazard analysis.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  solver.load<HazardAnalysis>(hazardManager);
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "failed to run hazard analysis";
    return signalPassFailure();
  }

  IRRewriter rewriter(op->getContext());

  // Collect the operations that required nop instructions.
  WalkResult result = op->walk([&](AMDGCNInstOpInterface instOp) {
    auto *afterState =
        solver.lookupState<HazardState>(solver.getProgramPointAfter(instOp));
    if (!afterState)
      return WalkResult::interrupt();

    InstCounts instCounts = afterState->flushedInstCounts;

    // Early return if no nop instructions are required.
    if (instCounts.isInactive())
      return WalkResult::advance();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(instOp);

    Location loc = instOp.getLoc();
    for (int i = 0; i < instCounts.getNumVector(); ++i)
      VNop::create(rewriter, loc);

    int8_t numScalarNops = instCounts.getNumScalar();

    // TODO: Don't hardcode 16 here, and get it from the NOP instruction.
    for (int32_t i = 0; i < numScalarNops; i += 16) {
      // The s_nop instruction has the quirk that if k is the number of wait
      // states in the instruction, then we see k + 1 wait states.
      int32_t nops = std::min(15, numScalarNops - i - 1);
      assert(nops >= 0 && "nops should be non-negative");
      inst::SOPPOp::create(rewriter, loc, OpCode::S_NOP,
                           static_cast<uint16_t>(nops));
    }

    assert(instCounts.getNumDataShare() == 0 &&
           "data share nop not implemented");
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    op->emitError() << "failed to resolve hazards";
    return signalPassFailure();
  }
}
