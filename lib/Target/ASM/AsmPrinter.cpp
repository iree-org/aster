//===- AsmPrinter.cpp - AMDGPU Assembly Printer -----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class for printing AMDGPU assembly.
//
//===----------------------------------------------------------------------===//

#include "aster/Target/ASM/AsmPrinter.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/ValueOrConst.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Print a comment line.
void aster::amdgcn::AsmPrinter::printComment(StringRef comment) {
  os << " ; " << comment << "\n";
}

static void printRegister(llvm::raw_ostream &os,
                          AMDGCNRegisterTypeInterface type) {
  RegisterRange range = type.getAsRange();
  StringRef prefix;
  switch (type.getRegisterKind()) {
  case RegisterKind::VGPR:
    prefix = "v";
    break;
  case RegisterKind::SGPR:
    prefix = "s";
    break;
  case RegisterKind::AGPR:
    prefix = "a";
    break;
  default:
    llvm_unreachable("nyi register kind");
  }
  os << " " << prefix;
  if (range.size() == 1) {
    os << range.begin().getRegister();
    return;
  }
  os << "[" << range.begin().getRegister() << ":"
     << range.end().getRegister() - 1 << "]";
}

void aster::amdgcn::AsmPrinter::printOperand(Value operand) {
  assert(instInProgress && "operand must be printed within an instruction");
  if (operand == nullptr) {
    os << " off";
    return;
  }
  llvm::APInt intValue;
  if (m_ConstantInt(&intValue).match(operand.getDefiningOp())) {
    os << " " << intValue;
    return;
  }
  llvm::APFloat floatValue(llvm::APFloat::IEEEdouble());
  if (m_ConstantFloat(&floatValue).match(operand.getDefiningOp())) {
    os << " " << floatValue;
    return;
  }
  printRegister(os, cast<AMDGCNRegisterTypeInterface>(operand.getType()));
}

void aster::amdgcn::AsmPrinter::printOffsetOperand(Value operand) {
  // This asserts if it's not a constant.
  int32_t value = *cast<ValueOrI32>(operand).getConst();
  if (value == 0)
    return;
  os << " offset: " << value;
}

void aster::amdgcn::AsmPrinter::printSquareIntModifier(StringRef modifier,
                                                       int64_t value,
                                                       int64_t defaultValue) {
  assert(instInProgress && "modifier must be printed within an instruction");
  if (value == defaultValue)
    return;
  os << " " << modifier << ":[" << value << "]";
}

void aster::amdgcn::AsmPrinter::printParenIntModifier(StringRef modifier,
                                                      int64_t value,
                                                      int64_t defaultValue) {
  assert(instInProgress && "modifier must be printed within an instruction");
  if (value == defaultValue)
    return;
  os << " " << modifier << "(" << value << ")";
}

void aster::amdgcn::AsmPrinter::printIntModifier(StringRef modifier,
                                                 int64_t value,
                                                 int64_t defaultValue) {
  assert(instInProgress && "modifier must be printed within an instruction");
  if (value == defaultValue)
    return;
  os << " " << modifier << ":" << value;
}

void aster::amdgcn::AsmPrinter::printIntModifier(int64_t value,
                                                 int64_t defaultValue) {
  assert(instInProgress && "modifier must be printed within an instruction");
  if (value == defaultValue)
    return;
  os << " " << value;
}

aster::amdgcn::AsmPrinter::PrintGuard
aster::amdgcn::AsmPrinter::printMnemonic(StringRef mnemonic) {
  os << mnemonic;
  return PrintGuard(*this);
}

void aster::amdgcn::AsmPrinter::endInstruction() {
  assert(instInProgress && "no instruction in progress to end");
  os << "\n";
  instInProgress = false;
}

std::string aster::amdgcn::AsmPrinter::getBranchLabel(Block *block) {
  if (blockLabels.contains(block)) {
    return blockLabels.lookup(block);
  }
  // Give correct name to entry block.
  if (block->isEntryBlock()) {
    // get KernelOp parent.
    Operation *parentOp = block->getParentOp();
    while (parentOp && !isa<KernelOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
    }
    assert(parentOp && "Entry block has no KernelOp parent");
    std::string label = cast<KernelOp>(parentOp).getName().str();
    blockLabels[block] = label;
    return label;
  }
  std::string label = ".AMDGCN_BB_" + std::to_string(blockLabels.size());
  blockLabels[block] = label;
  return label;
}

void aster::amdgcn::AsmPrinter::printBranchLabel(Block *block) {
  os << " " << getBranchLabel(block);
}
