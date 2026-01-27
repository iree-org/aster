//===- Utils.h - IR Utility Functions ---------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common IR utility functions used across ASTER.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_UTILS_H
#define ASTER_IR_UTILS_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <type_traits>

namespace mlir::aster {
/// Check if a value's defining block dominates a given block.
inline bool dominatesSuccessor(DominanceInfo &domInfo, Value value,
                               Block *block) {
  Block *defBlock = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    defBlock = blockArg.getOwner();
  else
    defBlock = value.getDefiningOp()->getBlock();
  return domInfo.properlyDominates(defBlock, block);
}

/// Check if a value dominates a given region successor.
inline bool dominatesSuccessor(DominanceInfo &domInfo, Value value,
                               RegionBranchOpInterface op,
                               RegionSuccessor successor) {
  if (successor.isParent())
    return domInfo.dominates(value, op);
  Block *defBlock = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    defBlock = blockArg.getOwner();
  else
    defBlock = value.getDefiningOp()->getBlock();
  return domInfo.properlyDominates(defBlock,
                                   &successor.getSuccessor()->front());
}

/// Walk all terminators in a region and invoke a function on each.
template <typename FuncTy, typename = std::enable_if_t<std::is_invocable_v<
                               FuncTy, RegionBranchTerminatorOpInterface>>>
inline void walkTerminators(Region *region, FuncTy &&func) {
  for (Block &block : *region) {
    if (block.empty())
      continue;
    if (auto terminator =
            dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
      func(terminator);
  }
}

} // namespace mlir::aster

#endif // ASTER_IR_UTILS_H
