//===- SSAMap.cpp - SSA value to ID mapping implementation -------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/IR/SSAMap.h"
#include "aster/IR/PrintingUtils.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::aster;

void SSAMap::insertOp(Operation *op) {
  bool added = false;
  for (Value result : op->getResults()) {
    if (valueToId.try_emplace(result, valueToId.size()).second)
      added = true;
  }
  if (added)
    irElementsWithValues.insert(op);
}

void SSAMap::insertBlock(Block *block) {
  bool added = false;
  for (BlockArgument arg : block->getArguments()) {
    if (valueToId.try_emplace(arg, valueToId.size()).second)
      added = true;
  }
  if (added)
    irElementsWithValues.insert(block);
}

int64_t SSAMap::getOrAddValue(Value value) {
  auto it = valueToId.find(value);
  if (it != valueToId.end())
    return it->second;

  if (BlockArgument blockArg = dyn_cast<BlockArgument>(value)) {
    insertBlock(blockArg.getOwner());
  } else if (Operation *op = value.getDefiningOp()) {
    insertOp(op);
  }

  it = valueToId.find(value);
  assert(it != valueToId.end() && "Value should have been inserted");
  return it->second;
}

void SSAMap::populateMap(Operation *op) {
  op->walk<WalkOrder::PreOrder>([this](Operation *visitedOp) {
    insertOp(visitedOp);
    for (Region &region : visitedOp->getRegions())
      for (Block &block : region)
        insertBlock(&block);
  });
}

void SSAMap::printMapMembers(llvm::raw_ostream &os) const {
  auto printer = [&](llvm::PointerUnion<Operation *, Block *> irElement) {
    if (Block *block = dyn_cast<Block *>(irElement)) {
      os << "Block: "
         << BlockWithFlags(block,
                           BlockWithFlags::PrintMode::PrintAsQualifiedOperand);
      os << "\n  arguments: [";
      llvm::interleaveComma(block->getArguments(), os,
                            [this, &os](BlockArgument arg) {
                              os << valueToId.lookup(arg) << " = `"
                                 << ValueWithFlags(arg, true) << "`";
                            });
      os << "]";
    } else {
      Operation *op = cast<Operation *>(irElement);
      os << "Operation: `" << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << "`";
      os << "\n  results: [";
      llvm::interleaveComma(op->getResults(), os, [this, &os](Value result) {
        os << valueToId.lookup(result) << " = `" << ValueWithFlags(result, true)
           << "`";
      });
      os << "]";
    }
  };
  llvm::interleave(irElementsWithValues, os, printer, "\n");
}

int64_t SSAMap::lookup(Value value) const {
  auto it = valueToId.find(value);
  return it != valueToId.end() ? it->second : -1;
}

bool SSAMap::contains(Value value) const { return valueToId.contains(value); }
