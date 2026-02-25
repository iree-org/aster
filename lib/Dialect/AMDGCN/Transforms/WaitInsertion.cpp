//===- WaitInsertion.cpp - Wait insertion using reaching definitions ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts wait operations using reaching definitions analysis. This
// is done by:
// 1. checking if any of the operands of an instruction belong to the current
// reaching definitions set.
// 2. if so, insert a wait operation for each of the reaching load operations
// before the instruction.
//
// Reaching definitions is configured to track only load operations. Futher,
// we provide a callback to kill reaching load operations if they are consumed
// as instruction inputs. The rationale is that if a load is consumed as an
// input, it means that the token for that load is dead. This allows us to not
// insert unnecessary wait operations.
//
// Example:
// clang-format off
// ```mlir
// %token = amdgcn.load global_load_dword dest %0 addr %4 offset c(%c0_i32) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
// amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
// amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
// ```
// After wait insertion:
// ```mlir
// %token = amdgcn.load global_load_dword dest %0 addr %3 offset c(%c0_i32) : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>, i32) -> !amdgcn.read_token<flat>
// memref.store %token, %alloca[] : memref<!amdgcn.read_token<flat>>
// %4 = memref.load %alloca[] : memref<!amdgcn.read_token<flat>>
// amdgcn.wait deps %4 : !amdgcn.read_token<flat>
// amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
// NOTE: No wait operation is inserted for the second test_inst, since the token is already consumed by the first test_inst.
// amdgcn.test_inst ins %0 : (!amdgcn.vgpr<?>) -> ()
// ```
// clang-format on
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/ReachingDefinitions.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_WAITINSERTION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// WaitInsertion pass
//===----------------------------------------------------------------------===//
struct WaitInsertion : public amdgcn::impl::WaitInsertionBase<WaitInsertion> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// WaitTransformImpl
//===----------------------------------------------------------------------===//
struct WaitTransformImpl {
  WaitTransformImpl(MLIRContext *ctx, DataFlowSolver &solver)
      : rewriter(ctx), solver(solver) {}

  /// Run the wait insertion transform.
  void run(FunctionOpInterface funcOp);

  /// Collect the reaching loads consumed by the given instruction.
  void collectDefinitions(InstOpInterface instOp);

  /// Handle the reaching loads consumed by the given instruction.
  void handleDefinitions(InstOpInterface instOp);

  IRRewriter rewriter;
  DataFlowSolver &solver;
  /// The entry block of the function.
  Block *entryBlock = nullptr;
  /// A map from load operations to their corresponding alloca operations.
  DenseMap<LoadOp, memref::AllocaOp> loadToAlloca;
  /// A set of load operations that are consumed by the current instruction.
  DenseSet<LoadOp> definitions;
};
} // namespace

void WaitTransformImpl::collectDefinitions(InstOpInterface instOp) {
  // Get the reaching definitions state before the instruction.
  const auto *reachingDefinitions =
      solver.lookupState<ReachingDefinitionsState>(
          solver.getProgramPointBefore(instOp));
  assert(reachingDefinitions &&
         "expected valid reaching definitions state before inst op");

  // Get the operands of the instruction.
  OperandRange outs = instOp.getInstOuts();
  OperandRange ins = instOp.getInstIns();
  int64_t outStartOperand = outs.empty() ? 0 : outs.getBeginOperandIndex();
  int64_t inStartOperand = ins.empty() ? 0 : ins.getBeginOperandIndex();
  MutableArrayRef<OpOperand> operands = instOp->getOpOperands();

  // Helper lambda to add definitions to the list.
  auto addDefinitions = [&](MutableArrayRef<OpOperand> operands) {
    for (OpOperand &operand : operands) {
      auto regTy = dyn_cast<RegisterTypeInterface>(operand.get().getType());
      if (!regTy || regTy.hasValueSemantics())
        continue;
      auto range = reachingDefinitions->getRange(operand.get());
      for (Definition definition : range)
        definitions.insert(cast<LoadOp>(definition.definition->getOwner()));
    }
  };

  // Collect the definitions for the output and input operands.
  if (!outs.empty())
    addDefinitions(operands.slice(outStartOperand, outs.size()));
  if (!ins.empty())
    addDefinitions(operands.slice(inStartOperand, ins.size()));
}

void WaitTransformImpl::handleDefinitions(InstOpInterface instOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(instOp);
  for (LoadOp loadOp : definitions) {
    memref::AllocaOp &allocaOp = loadToAlloca[loadOp];
    // Create the alloca operation if it doesn't exist.
    if (!allocaOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      // Create the alloca operation at the start of the entry block.
      rewriter.setInsertionPointToStart(entryBlock);
      allocaOp = memref::AllocaOp::create(
          rewriter, instOp.getLoc(),
          MemRefType::get({}, loadOp.getToken().getType()));

      // Store the token into the alloca operation.
      rewriter.setInsertionPointAfter(loadOp);
      memref::StoreOp::create(rewriter, loadOp.getLoc(), loadOp.getToken(),
                              allocaOp.getResult());
    }

    // Create the load operation.
    Value token = memref::LoadOp::create(rewriter, instOp.getLoc(),
                                         allocaOp.getResult(), ValueRange());
    WaitOp::create(rewriter, instOp.getLoc(), token);
  }
}

void WaitTransformImpl::run(FunctionOpInterface funcOp) {
  Region &body = funcOp.getFunctionBody();
  if (body.empty())
    return;
  entryBlock = &body.front();
  funcOp.walk([this](InstOpInterface instOp) {
    definitions.clear();
    collectDefinitions(instOp);
    handleDefinitions(instOp);
  });
}

void WaitInsertion::runOnOperation() {
  Operation *op = getOperation();

  // Create, configure, and run the data flow solver.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  dataflow::loadBaselineAnalyses(solver);
  auto loadFilter =
      +[](Operation *o) -> bool { return isa<amdgcn::LoadOp>(o); };

  // This callback has the effect of killing the loads if they are consumed as
  // input, the rationale being that this is equivalent to killing the token
  // definition, allowing us to not insert unnecessary wait operations.
  auto killCallback =
      +[](InstOpInterface instOp,
          ReachingDefinitionsAnalysis::KillDefsFn killDefs) -> LogicalResult {
    for (Value operand : instOp.getInstIns()) {
      assert((!isa<RegisterTypeInterface>(operand.getType()) ||
              !cast<RegisterTypeInterface>(operand.getType())
                   .hasValueSemantics()) &&
             "IR is not in post-to-register-semantics DPS normal form");

      // Get the allocas behind the operand.
      FailureOr<ValueRange> allocas = getAllocasOrFailure(operand);
      if (failed(allocas))
        return failure();
      killDefs(*allocas);
    }
    return success();
  };

  solver.load<ReachingDefinitionsAnalysis>(loadFilter, killCallback);
  if (failed(solver.initializeAndRun(op))) {
    op->emitError() << "Failed to run reaching definitions analysis";
    return signalPassFailure();
  }

  // Run the wait insertion transform for each function.
  op->walk([&](FunctionOpInterface funcOp) {
    WaitTransformImpl(op->getContext(), solver).run(funcOp);
  });
}
