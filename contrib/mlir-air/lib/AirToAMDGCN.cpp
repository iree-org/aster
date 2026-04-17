// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AirToAMDGCN.cpp - Lower AIR hierarchy ops to AMDGCN IR ------------===//
//
// Lowers air.launch, air.segment, air.herd, air.execute, and air.wait_all
// to flat AMDGCN-compatible IR:
//
//   air.launch IDs    -> gpu.block_id  (workgroup IDs)
//   air.herd tile IDs -> gpu.thread_id / 64  (wavefront index within workgroup)
//   air.segment       -> inline body (transparent wrapper)
//   air.execute       -> inline body (strip async)
//   air.wait_all     -> erase
//
// air.channel.put/get are not expected after air-to-amdgcn (air-dma-to-channel not used).
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map gpu::Dimension enum from an integer index.
static gpu::Dimension dimFromIndex(unsigned i) {
  switch (i) {
  case 0:
    return gpu::Dimension::x;
  case 1:
    return gpu::Dimension::y;
  case 2:
    return gpu::Dimension::z;
  default:
    llvm_unreachable("invalid dimension index");
  }
}

/// Clone all ops from `src` into `builder`'s insertion point, applying
/// `mapping`. Ops whose type matches any of `SkipOps...` are skipped.
template <typename... SkipOps>
static void cloneBodyOps(OpBuilder &builder, Block &src, IRMapping &mapping) {
  for (auto &op : src.getOperations()) {
    if ((isa<SkipOps>(op) || ...))
      continue;
    builder.clone(op, mapping);
  }
}

/// Strip async_dependencies from a channel op and drop its async_token result.
static void stripAsyncFromChannelOp(Operation *op) {
  if (auto asyncOp = dyn_cast<xilinx::air::AsyncOpInterface>(op)) {
    // Remove all async dependency operands.
    while (asyncOp.getAsyncDependencies().size() > 0)
      asyncOp.eraseAsyncDependency(0);
    // If the op has an async_token result, replace uses and drop it.
    if (auto token = asyncOp.getAsyncToken()) {
      token.replaceAllUsesWith(Value());
      // We can't actually remove a result from an existing op, but since all
      // token users will be erased (wait_all, other async deps), replacing
      // with null is sufficient — dead token uses are cleaned up below.
    }
  }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct AirToAMDGCN
    : public PassWrapper<AirToAMDGCN,
                         InterfacePass<aster::ModuleOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AirToAMDGCN)
  StringRef getArgument() const override { return "air-to-amdgcn"; }
  StringRef getDescription() const override {
    return "Lower AIR hierarchy ops to AMDGCN-compatible IR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    OpBuilder builder(moduleOp->getContext());

    // -----------------------------------------------------------------------
    // Phase 1: Strip async tokens.
    // -----------------------------------------------------------------------

    // Strip async from channel ops (they survive this pass).
    moduleOp->walk([&](xilinx::air::ChannelPutOp op) {
      stripAsyncFromChannelOp(op);
    });
    moduleOp->walk([&](xilinx::air::ChannelGetOp op) {
      stripAsyncFromChannelOp(op);
    });

    // Inline air.execute: splice body, replace results, erase.
    SmallVector<xilinx::air::ExecuteOp> executes;
    moduleOp->walk([&](xilinx::air::ExecuteOp op) { executes.push_back(op); });
    for (auto execOp : executes) {
      Block &body = execOp.getBody();
      auto terminator =
          cast<xilinx::air::ExecuteTerminatorOp>(body.getTerminator());

      // Replace non-token results (results 1..N) with terminator operands.
      for (unsigned i = 0; i < terminator.getNumOperands(); ++i)
        execOp.getResult(i + 1).replaceAllUsesWith(terminator.getOperand(i));

      // Replace async token result (result 0) — users will be erased.
      execOp.getResult(0).replaceAllUsesWith(Value());

      // Splice body ops (except terminator) before the execute op.
      auto &parentOps = execOp->getBlock()->getOperations();
      auto &bodyOps = body.getOperations();
      auto beforeTerminator = terminator->getIterator();
      parentOps.splice(execOp->getIterator(), bodyOps, bodyOps.begin(),
                       beforeTerminator);

      terminator->erase();
      execOp->erase();
    }

    // Erase air.wait_all ops.
    SmallVector<xilinx::air::WaitAllOp> waitAlls;
    moduleOp->walk(
        [&](xilinx::air::WaitAllOp op) { waitAlls.push_back(op); });
    for (auto op : waitAlls) {
      if (auto token = op.getAsyncToken())
        token.replaceAllUsesWith(Value());
      op->erase();
    }

    // -----------------------------------------------------------------------
    // Phase 2: Inline hierarchy ops (inside-out).
    // -----------------------------------------------------------------------

    // --- air.herd -> wavefront index (gpu.thread_id / wavefront_size) ---
    // Each herd tile is a wavefront (64 threads cooperating collectively).
    // Herd tile ID = wavefront index within the workgroup.
    SmallVector<xilinx::air::HerdOp> herds;
    moduleOp->walk([&](xilinx::air::HerdOp op) { herds.push_back(op); });
    for (auto herd : herds) {
      builder.setInsertionPoint(herd);
      Block &body = herd.getBody().front();
      unsigned numDims = herd.getNumDims();
      IRMapping mapping;

      // Map tile IDs to wavefront index = gpu.thread_id / 64.
      auto ids = herd.getIds();
      Value wavefrontSize = arith::ConstantIndexOp::create(
          builder, herd.getLoc(), 64);
      for (unsigned i = 0; i < numDims; ++i) {
        Value threadId = gpu::ThreadIdOp::create(builder, herd.getLoc(),
                                                 dimFromIndex(i));
        Value wavefrontId = arith::DivUIOp::create(builder, herd.getLoc(),
                                                    threadId, wavefrontSize);
        mapping.map(ids[i], wavefrontId);
      }

      // Map tile sizes to size operands.
      auto sizeArgs = herd.getSize();
      auto sizeOperands = herd.getSizeOperands();
      for (unsigned i = 0; i < numDims; ++i)
        mapping.map(sizeArgs[i], sizeOperands[i]);

      // Map kernel arguments to kernel operands.
      auto kernelArgs = herd.getKernelArguments();
      auto kernelOperands = herd.getKernelOperands();
      for (unsigned i = 0; i < kernelArgs.size(); ++i)
        mapping.map(kernelArgs[i], kernelOperands[i]);

      // Clone body.
      cloneBodyOps<xilinx::air::HerdTerminatorOp>(builder, body, mapping);

      // Replace async token if present.
      if (auto token = herd.getAsyncToken())
        token.replaceAllUsesWith(Value());
      herd->erase();
    }

    // --- air.segment -> inline ---
    SmallVector<xilinx::air::SegmentOp> segments;
    moduleOp->walk(
        [&](xilinx::air::SegmentOp op) { segments.push_back(op); });
    for (auto segment : segments) {
      builder.setInsertionPoint(segment);
      Block &body = segment.getBody().front();
      unsigned numDims = segment.getNumDims();
      IRMapping mapping;

      // Map segment IDs (if any) to gpu.block_id ops.
      auto ids = segment.getIds();
      auto sizeArgs = segment.getSize();
      auto sizeOperands = segment.getSizeOperands();
      for (unsigned i = 0; i < numDims; ++i) {
        Value blockId = gpu::BlockIdOp::create(builder, segment.getLoc(),
                                               dimFromIndex(i));
        mapping.map(ids[i], blockId);
        mapping.map(sizeArgs[i], sizeOperands[i]);
      }

      // Map kernel arguments to kernel operands.
      auto kernelArgs = segment.getKernelArguments();
      auto kernelOperands = segment.getKernelOperands();
      for (unsigned i = 0; i < kernelArgs.size(); ++i)
        mapping.map(kernelArgs[i], kernelOperands[i]);

      cloneBodyOps<xilinx::air::SegmentTerminatorOp>(builder, body, mapping);

      if (auto token = segment.getAsyncToken())
        token.replaceAllUsesWith(Value());
      segment->erase();
    }

    // --- air.launch -> gpu.block_id ---
    SmallVector<xilinx::air::LaunchOp> launches;
    moduleOp->walk(
        [&](xilinx::air::LaunchOp op) { launches.push_back(op); });
    for (auto launch : launches) {
      builder.setInsertionPoint(launch);
      Block &body = launch.getBody().front();
      unsigned numDims = launch.getNumDims();
      IRMapping mapping;

      // Map launch IDs to gpu.block_id ops.
      auto ids = launch.getIds();
      auto sizeArgs = launch.getSize();
      auto sizeOperands = launch.getSizeOperands();
      for (unsigned i = 0; i < numDims; ++i) {
        Value blockId = gpu::BlockIdOp::create(builder, launch.getLoc(),
                                               dimFromIndex(i));
        mapping.map(ids[i], blockId);
        mapping.map(sizeArgs[i], sizeOperands[i]);
      }

      // Map kernel arguments to kernel operands.
      auto kernelArgs = launch.getKernelArguments();
      auto kernelOperands = launch.getKernelOperands();
      for (unsigned i = 0; i < kernelArgs.size(); ++i)
        mapping.map(kernelArgs[i], kernelOperands[i]);

      cloneBodyOps<xilinx::air::LaunchTerminatorOp>(builder, body, mapping);

      if (auto token = launch.getAsyncToken())
        token.replaceAllUsesWith(Value());
      launch->erase();
    }

    // --- scf.parallel -> wavefront ID inlining ---
    // After air-dma-to-channel, hoisted channel.put/get ops are wrapped in
    // scf.parallel loops that iterate over the herd tile space. These must
    // be inlined: replace induction variables with gpu.thread_id / 64
    // (wavefront index), then splice body ops in place of the parallel.
    SmallVector<scf::ParallelOp> parallels;
    moduleOp->walk(
        [&](scf::ParallelOp op) { parallels.push_back(op); });
    for (auto parallel : parallels) {
      builder.setInsertionPoint(parallel);
      Location loc = parallel.getLoc();
      Block &body = parallel.getRegion().front();

      // Compute wavefront ID for each induction variable.
      // All dimensions are derived from thread_id_x (1D wavefront layout).
      Value wavefrontSize =
          arith::ConstantIndexOp::create(builder, loc, 64);
      Value threadIdX =
          gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
      Value wavefrontId =
          arith::DivUIOp::create(builder, loc, threadIdX, wavefrontSize);

      auto ivs = parallel.getInductionVars();
      if (ivs.size() == 1) {
        // 1D parallel: iv = wavefrontId directly.
        ivs[0].replaceAllUsesWith(wavefrontId);
      } else if (ivs.size() == 2) {
        // 2D parallel: decompose wavefrontId into (x, y).
        auto ub = parallel.getUpperBound();
        Value ubX = ub[0];
        Value ivX = arith::RemUIOp::create(builder, loc, wavefrontId, ubX);
        Value ivY = arith::DivUIOp::create(builder, loc, wavefrontId, ubX);
        ivs[0].replaceAllUsesWith(ivX);
        ivs[1].replaceAllUsesWith(ivY);
      }

      // Replace init values / results. scf.parallel with reduce produces
      // results from scf.reduce. Replace with init values (no actual
      // reduction needed — each wavefront runs independently).
      for (unsigned i = 0; i < parallel.getNumResults(); ++i)
        parallel.getResult(i).replaceAllUsesWith(parallel.getInitVals()[i]);

      // Splice body ops (skip the yield/reduce terminator) before parallel.
      auto &parentOps = parallel->getBlock()->getOperations();
      auto &bodyOps = body.getOperations();
      // Find the last non-terminator op.
      auto termIt = body.getTerminator()->getIterator();
      parentOps.splice(parallel->getIterator(), bodyOps, bodyOps.begin(),
                       termIt);

      parallel->erase();
    }
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createAirToAMDGCN() {
  return std::make_unique<AirToAMDGCN>();
}
} // namespace mlir::aster::mlir_air
