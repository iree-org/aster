//===- ExpandMetadataOps.cpp ----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <type_traits>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_EXPANDMETADATAOPS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// ExpandMetadataOps pass
//===----------------------------------------------------------------------===//
struct ExpandMetadataOps
    : public amdgcn::impl::ExpandMetadataOpsBase<ExpandMetadataOps> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static Value loadArgument(RewriterBase &rewriter, Value kenArgPtr, Value alloc,
                          uint32_t size, int32_t offset) {
  llvm::function_ref<Value(OpBuilder &, Location, Value, Value, uint32_t)>
      createOp;
  RegisterTypeInterface loadTy{};
  int32_t numWords;
  uint32_t szWordsFloor = size / 4;
  // Determine the best load instruction to use.
  if (szWordsFloor % 16 == 0) {
    numWords = 16;
    loadTy =
        rewriter.getType<SGPRRangeType>(RegisterRange(Register(), numWords));
    createOp = S_LOAD_DWORDX16::create;
  } else if (szWordsFloor % 8 == 0) {
    numWords = 8;
    loadTy =
        rewriter.getType<SGPRRangeType>(RegisterRange(Register(), numWords));
    createOp = S_LOAD_DWORDX8::create;
  } else if (szWordsFloor % 4 == 0) {
    numWords = 4;
    loadTy =
        rewriter.getType<SGPRRangeType>(RegisterRange(Register(), numWords));
    createOp = S_LOAD_DWORDX4::create;
  } else if (szWordsFloor % 2 == 0) {
    numWords = 2;
    loadTy =
        rewriter.getType<SGPRRangeType>(RegisterRange(Register(), numWords));
    createOp = S_LOAD_DWORDX2::create;
  } else {
    numWords = 1;
    loadTy = rewriter.getType<SGPRType>(Register());
    createOp = S_LOAD_DWORD::create;
  }
  int32_t numLoads = ((size + 3) / 4) / numWords;

  // Load the easy case.
  if (numLoads == 1)
    return createOp(rewriter, alloc.getLoc(), alloc, kenArgPtr, offset);

  // Load in multiple instructions.
  ValueRange splitAlloc = splitRange(rewriter, alloc.getLoc(), alloc);
  SmallVector<Value> loadedRegs;
  for (int32_t i = 0; i < numLoads; ++i) {
    Value dest;
    // Get the destination.
    if (numWords > 1) {
      dest = MakeRegisterRangeOp::create(
          rewriter, alloc.getLoc(), splitAlloc.slice(i * numWords, numWords));
    } else {
      dest = splitAlloc[i];
    }

    // Load the segment.
    Value segment = createOp(rewriter, alloc.getLoc(), dest, kenArgPtr,
                             offset + i * 4 * numWords);

    // Maybe partition the segment.
    if (numWords > 1) {
      llvm::append_range(loadedRegs,
                         splitRange(rewriter, alloc.getLoc(), segment));
    } else {
      loadedRegs.push_back(segment);
    }
  }
  return MakeRegisterRangeOp::create(rewriter, alloc.getLoc(), loadedRegs);
}

/// Handle the LoadArgOps in a kernel.
static LogicalResult handleArgs(RewriterBase &rewriter, KernelOp op,
                                ArrayRef<LoadArgOp> ops) {
  ArrayRef<KernelArgAttrInterface> args = op.getArguments();
  int32_t offset = op.getEnablePrivateSegmentBuffer() ? 4 : 0;
  offset += op.getEnableDispatchPtr() ? 2 : 0;
  KernelArgSegmentInfo argInfo = KernelArgSegmentInfo::get(op);
  // TODO: handle the queue ptr arguments as well.
  // Get the entry block.
  Block *entry = &op.getBodyRegion().front();
  rewriter.setInsertionPointToStart(entry);
  // Get the alloca for the kernel arguments.
  Value kenArgPtr = createAllocation(
      rewriter, op.getLoc(),
      amdgcn::SGPRRangeType::get(rewriter.getContext(),
                                 RegisterRange(Register(offset), 2)));
  rewriter.setInsertionPointAfter(kenArgPtr.getDefiningOp());

  // Handle each LoadArgOp.
  for (LoadArgOp arg : ops) {
    // Set insertion point to the LoadArgOp.
    rewriter.setInsertionPoint(arg);

    int64_t index = arg.getIndex();
    // This should be guaranteed by verification, but check it anyway.
    if (static_cast<int64_t>(args.size()) <= index || index < 0) {
      return arg.emitError("argument index out of bounds");
    }

    // Get the argument attribute.
    KernelArgAttrInterface argAttr = args[index];
    uint32_t size = argAttr.getSize();
    assert(size >= 4 && "expected argument size greater than 4 bytes");

    // Create the allocation for the loaded argument.
    Value alloc = createAllocation(
        rewriter, arg.getLoc(),
        amdgcn::SGPRRangeType::get(rewriter.getContext(),
                                   RegisterRange(Register(), (size + 3) / 4)));
    // Load the argument from the kernel argument pointer.
    Value loadedArg =
        loadArgument(rewriter, kenArgPtr, alloc, size, argInfo.offsets[index]);
    // Replace the LoadArgOp with the loaded argument.
    rewriter.replaceOp(arg, loadedArg);
  }
  return success();
}

/// Handle the BlockIdOps in a kernel.
static void handleBlockId(RewriterBase &rewriter, KernelOp op,
                          ArrayRef<BlockIdOp> ops) {
  // Get the entry block.
  Block *entry = &op.getBodyRegion().front();
  rewriter.setInsertionPointToStart(entry);

  int32_t offset = op.getEnablePrivateSegmentBuffer() ? 4 : 0;
  offset += op.getEnableKernargSegmentPtr() ? 2 : 0;
  offset += op.getEnableDispatchPtr() ? 2 : 0;

  // Handle each block id.
  for (BlockIdOp blockId : ops) {
    int32_t dim = static_cast<int32_t>(blockId.getDim());
    Value id = createAllocation(
        rewriter, blockId.getLoc(),
        SGPRType::get(rewriter.getContext(), Register(offset + dim)));
    rewriter.replaceOp(blockId, id);
  }
}

/// Handle the ThreadIdOps in a kernel.
static void handleThreadId(RewriterBase &rewriter, KernelOp op,
                           ArrayRef<ThreadIdOp> ops) {
  // Get the entry block.
  Block *entry = &op.getBodyRegion().front();
  rewriter.setInsertionPointToStart(entry);

  // Handle each thread id.
  for (ThreadIdOp threadId : ops) {
    int32_t dim = static_cast<int32_t>(threadId.getDim());
    Value id =
        createAllocation(rewriter, threadId.getLoc(),
                         VGPRType::get(rewriter.getContext(), Register(dim)));
    rewriter.replaceOp(threadId, id);
  }
}

template <typename DimOp>
static void handledDim(RewriterBase &rewriter, KernelOp op,
                       SmallVectorImpl<LoadArgOp> &loadArgs,
                       ArrayRef<DimOp> ops, ArrayRef<bool> dimSeen) {
  using ArgAttr = std::conditional_t<std::is_same_v<DimOp, GridDimOp>,
                                     GridDimArgAttr, BlockDimArgAttr>;
  // Get the entry block.
  Block *entry = &op.getBodyRegion().front();
  rewriter.setInsertionPointToStart(entry);

  std::array<int32_t, 3> dimIndex = {-1, -1, -1};
  // Get the arguments.
  SmallVector<KernelArgAttrInterface> args;
  llvm::append_range(args, op.getArguments());
  bool modified = false;
  for (int32_t d = 0; d < 3; ++d) {
    // Skip unused dimensions.
    if (!dimSeen[d])
      continue;
    auto attr = ArgAttr::get(op.getContext(), static_cast<Dim>(d));
    auto it = llvm::find(args, attr);

    // Add the argument if not present.
    if (it == args.end()) {
      dimIndex[d] = static_cast<int32_t>(args.size());
      args.push_back(attr);
      modified = true;
    } else {
      dimIndex[d] = static_cast<int32_t>(std::distance(args.begin(), it));
    }
  }
  // Update the arguments if modified.
  if (modified)
    op.setArguments(args);

  // Handle each dim op.
  for (DimOp dimOp : ops) {
    int32_t dim = static_cast<int32_t>(dimOp.getDim());
    LoadArgOp lOp = LoadArgOp::create(rewriter, dimOp.getLoc(), dimOp.getType(),
                                      dimIndex[dim]);
    Value value = lOp.getResult();
    if constexpr (std::is_same_v<DimOp, BlockDimOp>) {
      Value alloca =
          createAllocation(rewriter, dimOp.getLoc(),
                           SGPRType::get(rewriter.getContext(), Register()));
      Value cMagic = arith::ConstantOp::create(
          rewriter, dimOp.getLoc(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), 0xFFFF));
      // TODO: remove this and let the optimizer handle it.
      S_WAITCNT::create(rewriter, dimOp.getLoc());
      value = S_AND_B32::create(rewriter, value.getLoc(), alloca, value, cMagic)
                  .getSdstRes();
    }
    rewriter.replaceOp(dimOp, value);
    loadArgs.push_back(lOp);
  }
}

//===----------------------------------------------------------------------===//
// ExpandMetadataOps pass
//===----------------------------------------------------------------------===//

void ExpandMetadataOps::runOnOperation() {
  KernelOp op = getOperation();
  // Collect all relevant ops.
  SmallVector<LoadArgOp> loadArgs;
  SmallVector<ThreadIdOp> threadIds;
  SmallVector<BlockDimOp> blockDims;
  SmallVector<BlockIdOp> blockIds;
  SmallVector<GridDimOp> gridDims;
  std::array<bool, 3> threadIdSeen = {false, false, false};
  std::array<bool, 3> blockIdSeen = {false, false, false};
  std::array<bool, 3> blockDimSeen = {false, false, false};
  std::array<bool, 3> gridDimSeen = {false, false, false};
  op.walk([&](Operation *op) {
    if (auto arg = dyn_cast<LoadArgOp>(op)) {
      loadArgs.push_back(arg);
    } else if (auto threadId = dyn_cast<ThreadIdOp>(op)) {
      int32_t dim = static_cast<int32_t>(threadId.getDim());
      threadIds.push_back(threadId);
      threadIdSeen[dim] = true;
    } else if (auto blockDim = dyn_cast<BlockDimOp>(op)) {
      int32_t dim = static_cast<int32_t>(blockDim.getDim());
      blockDims.push_back(blockDim);
      blockDimSeen[dim] = true;
    } else if (auto blockId = dyn_cast<BlockIdOp>(op)) {
      int32_t dim = static_cast<int32_t>(blockId.getDim());
      blockIds.push_back(blockId);
      blockIdSeen[dim] = true;
    } else if (auto gridDim = dyn_cast<GridDimOp>(op)) {
      int32_t dim = static_cast<int32_t>(gridDim.getDim());
      gridDims.push_back(gridDim);
      gridDimSeen[dim] = true;
    }
  });

  // Handle the arguments.
  IRRewriter rewriter(op);
  handledDim<BlockDimOp>(rewriter, op, loadArgs, blockDims, blockDimSeen);
  handledDim<GridDimOp>(rewriter, op, loadArgs, gridDims, gridDimSeen);
  if (loadArgs.size() > 0 && failed(handleArgs(rewriter, op, loadArgs)))
    return signalPassFailure();

  // Handle the block and thread ids.
  op.setEnableWorkgroupIdX(blockIdSeen[0]);
  op.setEnableWorkgroupIdY(blockIdSeen[1]);
  op.setEnableWorkgroupIdZ(blockIdSeen[2]);
  if (threadIdSeen[2])
    op.setWorkitemIdMode(WorkitemIDMode::XYZ);
  else if (threadIdSeen[1])
    op.setWorkitemIdMode(WorkitemIDMode::XY);
  else if (threadIdSeen[0])
    op.setWorkitemIdMode(WorkitemIDMode::X);

  handleBlockId(rewriter, op, blockIds);
  handleThreadId(rewriter, op, threadIds);
}
