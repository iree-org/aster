//===- AsmPrinter.h - AMDGPU Assembly Printer -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AsmPrinter class for printing AMDGPU assembly.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TARGET_ASM_ASMPRINTER_H
#define ASTER_TARGET_ASM_ASMPRINTER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Value;
class Block;
} // namespace mlir

namespace mlir::aster {
namespace amdgcn {
/// Class for printing AMDGPU assembly code.
class AsmPrinter {
public:
  /// Construct an AsmPrinter with the given output stream.
  explicit AsmPrinter(llvm::raw_ostream &os) : os(os) {}

  /// Get the output stream.
  llvm::raw_ostream &getStream() { return os; }
  /// Print a comment line.
  void printComment(StringRef comment);
  /// Print a comma separator.
  void printComma() {
    assert(instInProgress && "comma must be printed within an instruction");
    os << ",";
  }
  /// Print a keyword.
  void printKeyword(StringRef keyword) {
    assert(instInProgress && "keyword must be printed within an instruction");
    os << " " << keyword;
  }
  /// Print an operand.
  void printOperand(Value operand);
  /// Print an operand.
  void printOffsetOperand(Value operand);
  /// Print a modifier of the form:
  /// modifier:[value]?
  void printSquareIntModifier(StringRef modifier, int64_t value,
                              int64_t defaultValue);
  /// Print a modifier of the form:
  /// modifier:value?
  void printIntModifier(StringRef modifier, int64_t value,
                        int64_t defaultValue);
  /// Print a modifier of the form:
  /// value?
  void printIntModifier(int64_t value, int64_t defaultValue);
  /// Print a modifier of the form:
  /// name(value)?
  void printParenIntModifier(StringRef modifier, int64_t value,
                             int64_t defaultValue);

  /// RAII guard for printing an instruction.
  struct PrintGuard {
    PrintGuard(AsmPrinter &printer) : printer(printer) {
      printer.startInstruction();
    }
    ~PrintGuard() { printer.endInstruction(); }

  private:
    AsmPrinter &printer;
  };

  /// Print the mnemonic.
  PrintGuard printMnemonic(StringRef mnemonic);
  /// Prints a label for a block.
  std::string getBranchLabel(Block *block);
  void printBranchLabel(Block *block);

private:
  friend struct PrintGuard;
  // Start and end instruction printing.
  void startInstruction() {
    assert(!instInProgress && "instruction already in progress");
    instInProgress = true;
  }
  void endInstruction();
  /// The output stream.
  llvm::raw_ostream &os;
  /// Whether an instruction is currently being printed.
  bool instInProgress = false;
  llvm::DenseMap<Block *, std::string> blockLabels;
};
} // namespace amdgcn
} // namespace mlir::aster

#endif // ASTER_TARGET_ASM_ASMPRINTER_H
