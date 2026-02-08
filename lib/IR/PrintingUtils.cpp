//===- PrintingUtils.cpp - IR Printing Utilities Implementation --*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/IR/PrintingUtils.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::aster;

llvm::raw_ostream &mlir::aster::operator<<(llvm::raw_ostream &os,
                                           const BlockWithFlags &bwf) {
  if (bwf.getBlock() == nullptr) {
    os << "<<NULL BLOCK>>";
    return os;
  }

  // Check if the block is unlinked.
  Operation *parentOp = bwf.block->getParentOp();
  if (parentOp == nullptr) {
    os << "<<UNLINKED BLOCK>>";
    return os;
  }

  Region *region = bwf.block->getParent();
  assert(region && "Block should have a parent region");

  switch (bwf.getPrintMode()) {
  case BlockWithFlags::PrintMode::PrintAsOperand: {
    AsmState state(parentOp, bwf.theFlags);
    bwf.block->printAsOperand(os, state);
    return os;
  }
  case BlockWithFlags::PrintMode::PrintAsQualifiedOperand: {
    OpPrintingFlags flags = bwf.theFlags;
    AsmState state(parentOp, bwf.theFlags);
    os << "Block<op = " << OpWithFlags(parentOp, flags.skipRegions(true));
    os << ", region = " << region->getRegionNumber();
    os << ", bb = ";
    bwf.block->printAsOperand(os, state);
    os << ", args = [";
    llvm::interleaveComma(
        bwf.block->getArguments(), os,
        [&](BlockArgument arg) { os << ValueWithFlags(arg, true, flags); });
    os << "]>";
    return os;
  }
  default:
    break;
  }
  // Print the block as a regular block.
  AsmState state(parentOp, bwf.theFlags);
  bwf.block->print(os, state);
  return os;
}
