//===- LDSMultibufferPrep.cpp - Pre-pipelining LDS multi-buffer prep ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepares LDS buffers for multi-buffering before the SCF pipeliner runs.
// Reads sched.stage annotations on alloc_lds/dealloc_lds pairs, hoists N
// copies before the loop, and replaces in-loop usage with rotating iter_args.
//
// This is a standalone transformation called from the pipelining pass. A test
// pass wrapper (TestLDSMultibufferPrepPass.cpp) exists for unit testing.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Transforms.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::aster;

/// Helper to read sched.stage from an operation, returns -1 if absent.
static int getStage(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("sched.stage"))
    return attr.getInt();
  return -1;
}

namespace {

/// LDS buffer group spanning multiple pipeline stages.
struct LDSGroup {
  amdgcn::AllocLDSOp allocOp;
  amdgcn::DeallocLDSOp deallocOp;
  amdgcn::GetLDSOffsetOp getOffsetOp;
  int allocStage;
  int deallocStage;
  /// Includes both alloc and dealloc stages in the count.
  int numBuffers() const { return deallocStage - allocStage + 1; }
};

/// Find LDS groups (alloc/dealloc pairs with sched.stage) in a loop body.
static SmallVector<LDSGroup> findLDSGroups(scf::ForOp forOp) {
  SmallVector<LDSGroup> groups;

  forOp.getBody()->walk([&](amdgcn::AllocLDSOp allocOp) {
    int allocStage = getStage(allocOp);
    if (allocStage < 0)
      return;

    // Find the matching dealloc_lds.
    amdgcn::DeallocLDSOp deallocOp;
    amdgcn::GetLDSOffsetOp getOffsetOp;
    for (Operation *user : allocOp.getBuffer().getUsers()) {
      if (auto dealloc = dyn_cast<amdgcn::DeallocLDSOp>(user))
        deallocOp = dealloc;
      else if (auto getOff = dyn_cast<amdgcn::GetLDSOffsetOp>(user))
        getOffsetOp = getOff;
    }

    if (!deallocOp) {
      allocOp->emitWarning(
          "alloc_lds with sched.stage but no matching dealloc_lds");
      return;
    }
    if (!getOffsetOp) {
      allocOp->emitWarning("alloc_lds with sched.stage but no get_lds_offset");
      return;
    }

    int deallocStage = getStage(deallocOp);
    if (deallocStage < allocStage) {
      allocOp->emitWarning()
          << "dealloc_lds stage (" << deallocStage
          << ") is before alloc_lds stage (" << allocStage << ")";
      return;
    }

    groups.push_back(
        {allocOp, deallocOp, getOffsetOp, allocStage, deallocStage});
  });

  return groups;
}

/// Transform a single loop: hoist LDS allocs and add rotating offset iter_args.
static LogicalResult transformLoop(scf::ForOp forOp,
                                   SmallVectorImpl<LDSGroup> &groups) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Hoist N alloc_lds + get_lds_offset pairs before the loop.
  struct HoistedGroup {
    SmallVector<amdgcn::AllocLDSOp> allocs;
    SmallVector<Value> offsets; // results of get_lds_offset
  };
  SmallVector<HoistedGroup> hoisted;

  for (LDSGroup &g : groups) {
    int N = g.numBuffers();
    HoistedGroup hg;

    for (int i = 0; i < N; ++i) {
      auto newAlloc = amdgcn::AllocLDSOp::create(
          builder, loc, g.allocOp.getDynamicSize(), g.allocOp.getStaticSize(),
          g.allocOp.getAlignment(),
          /*offset=*/IntegerAttr{});
      auto newGetOff = amdgcn::GetLDSOffsetOp::create(
          builder, loc, g.getOffsetOp.getResult().getType(),
          newAlloc.getBuffer());
      hg.allocs.push_back(newAlloc);
      hg.offsets.push_back(newGetOff.getResult());
    }
    hoisted.push_back(std::move(hg));
  }

  // Collect init values: existing iter_args first, then LDS offsets.
  int64_t numExisting = forOp.getNumRegionIterArgs();
  SmallVector<Value> newIterArgs;
  for (Value init : forOp.getInits())
    newIterArgs.push_back(init);
  for (const HoistedGroup &hg : hoisted) {
    for (Value off : hg.offsets)
      newIterArgs.push_back(off);
  }

  // Rebuild the loop with combined iter_args.
  auto newForOp =
      scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newIterArgs);

  // Map old induction var to new one.
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // Map existing iter_args (identity, same position).
  for (int64_t i = 0; i < numExisting; ++i)
    mapping.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);

  // Compute base iter_arg index for each LDS group.
  SmallVector<int64_t> groupIterArgBase;
  int64_t idx = numExisting;
  for (int64_t gi = 0, gie = groups.size(); gi < gie; ++gi) {
    groupIterArgBase.push_back(idx);
    idx += hoisted[gi].offsets.size();
  }

  // Clone loop body, skipping LDS ops (already hoisted/replaced).
  // For each cloned op, map LDS offsets to stage-appropriate iter_args.
  OpBuilder bodyBuilder(forOp.getContext());
  bodyBuilder.setInsertionPointToEnd(newForOp.getBody());
  SmallPtrSet<Operation *, 8> opsToSkip;
  for (LDSGroup &g : groups) {
    opsToSkip.insert(g.allocOp);
    opsToSkip.insert(g.deallocOp);
    opsToSkip.insert(g.getOffsetOp);
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (opsToSkip.contains(&op))
      continue;

    // Map each LDS offset to a stage-appropriate iter_arg position.
    //
    // The yield rotates offsets left by 1 each iteration. In the pipelined
    // kernel, stage S processes an iteration whose data was loaded S-allocStage
    // iterations earlier. After that many left-rotations, the buffer that was
    // at position 0 is now at position N - stageDelta.
    // So stage S reads from position N - stageDelta.
    int opStage = getStage(&op);
    for (int64_t gi = 0, gie = groups.size(); gi < gie; ++gi) {
      LDSGroup &g = groups[gi];
      int N = g.numBuffers();
      int position = 0;
      if (opStage >= 0 && opStage > g.allocStage && opStage <= g.deallocStage) {
        // Op is within this group's stage span. Compute which rotated
        // buffer position it should read from.
        int stageDelta = opStage - g.allocStage;
        position = N - stageDelta;
      }
      // Ops outside the group's stage range (or with no stage annotation)
      // default to position 0. If they don't reference this group's offset
      // the mapping won't be consulted.
      mapping.map(
          g.getOffsetOp.getResult(),
          newForOp.getRegionIterArgs()[groupIterArgBase[gi] + position]);
    }

    bodyBuilder.clone(op, mapping);
  }

  // Yield: [existing yield operands (remapped), rotated LDS offsets].
  SmallVector<Value> yieldValues;

  // Clone existing yield operands from original loop.
  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (Value yieldVal : oldYield.getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(yieldVal));

  // Rotated LDS offsets: [off1, ..., off_{N-1}, off0] per group.
  for (int64_t gi = 0, gie = groups.size(); gi < gie; ++gi) {
    int N = groups[gi].numBuffers();
    int64_t base = groupIterArgBase[gi];
    SmallVector<Value> groupArgs;
    for (int i = 0; i < N; ++i)
      groupArgs.push_back(newForOp.getRegionIterArgs()[base + i]);
    // Rotate left by 1: [1, 2, ..., N-1, 0]
    for (int i = 1; i < N; ++i)
      yieldValues.push_back(groupArgs[i]);
    yieldValues.push_back(groupArgs[0]);
  }

  // Create yield with rotated offsets.
  scf::YieldOp::create(bodyBuilder, loc, yieldValues);

  // Place dealloc_lds for all hoisted buffers after the loop.
  builder.setInsertionPointAfter(newForOp);
  for (const HoistedGroup &hg : hoisted) {
    for (auto allocOp : hg.allocs)
      amdgcn::DeallocLDSOp::create(builder, loc, allocOp.getBuffer());
  }

  // Replace uses of old loop results with corresponding new loop results.
  for (int64_t i = 0; i < numExisting; ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  // Erase old loop.
  forOp.erase();

  return success();
}

} // namespace

LogicalResult mlir::aster::prepareLDSMultibuffers(Operation *op) {
  // Collect loops first to avoid iterator invalidation.
  SmallVector<scf::ForOp> loops;
  op->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    auto groups = findLDSGroups(forOp);
    if (groups.empty())
      continue;
    if (failed(transformLoop(forOp, groups)))
      return failure();
  }
  return success();
}
