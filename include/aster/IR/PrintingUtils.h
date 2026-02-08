//===- PrintingUtils.h - IR Printing Utilities -----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares printing utilities for MLIR IR.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_PRINTING_UTILS_H
#define ASTER_IR_PRINTING_UTILS_H

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster {
//===----------------------------------------------------------------------===//
// ValueWithFlags
//===----------------------------------------------------------------------===//

/// A wrapper class that allows for printing a Value with a set of flags,
/// useful as a stream modifier similar to OpWithFlags.
/// Example:
/// `llvm::dbgs() << ValueWithFlags(value, OpPrintingFlags().skipRegions());`
class ValueWithFlags {
public:
  ValueWithFlags(Value value, bool printAsOperand = false,
                 OpPrintingFlags flags = {})
      : value(value), theFlags(flags), printAsOperand(printAsOperand) {}
  OpPrintingFlags &flags() { return theFlags; }
  const OpPrintingFlags &flags() const { return theFlags; }
  bool getPrintAsOperand() const { return printAsOperand; }
  void setPrintAsOperand(bool printAsOperand) {
    this->printAsOperand = printAsOperand;
  }
  Value getValue() const { return value; }

private:
  Value value;
  OpPrintingFlags theFlags;
  bool printAsOperand = false;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const ValueWithFlags &vwf);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ValueWithFlags &vwf) {
  if (vwf.getPrintAsOperand()) {
    vwf.value.printAsOperand(os, vwf.theFlags);
    return os;
  }
  vwf.value.print(os, vwf.theFlags);
  return os;
}

//===----------------------------------------------------------------------===//
// BlockWithFlags
//===----------------------------------------------------------------------===//

/// A wrapper class that allows for printing a Block with a set of flags,
/// useful as a stream modifier similar to OpWithFlags.
/// Example:
/// `llvm::dbgs() << BlockWithFlags(block, OpPrintingFlags().skipRegions());`
class BlockWithFlags {
public:
  enum class PrintMode {
    Default,
    PrintAsOperand,
    PrintAsQualifiedOperand,
  };
  BlockWithFlags(Block *block, PrintMode printMode = PrintMode::Default,
                 OpPrintingFlags flags = {})
      : block(block), theFlags(flags), printMode(printMode) {}
  OpPrintingFlags &flags() { return theFlags; }
  const OpPrintingFlags &flags() const { return theFlags; }
  PrintMode getPrintMode() const { return printMode; }
  void setPrintMode(PrintMode printMode) { this->printMode = printMode; }
  Block *getBlock() const { return block; }

private:
  Block *block;
  OpPrintingFlags theFlags;
  PrintMode printMode;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const BlockWithFlags &bwf);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BlockWithFlags &bwf);

} // namespace mlir::aster

#endif // ASTER_IR_PRINTING_UTILS_H
