//===- AMDGCNInstAsmPrinter.cpp - Export ASM --------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen-generated ISA instruction printers. Kept in a separate translation
// unit so edits to TranslateModule.cpp do not recompile ~8k lines of generated
// code.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Target/ASM/AsmPrinter.h"
#include "aster/Target/ASM/TranslateModule.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::amdgcn::target;

/// Print offset operand, which is either a constant or absent (zero).
static void printLSOffset(amdgcn::AsmPrinter &printer, AMDGCNInstOpInterface op,
                          Value off, Value cOff) {
  if (off && !cOff) {
    printer.printOperand(off);
    return;
  }
  if (off && cOff) {
    printer.printOperand(off);
    printer.printOffsetOperand(cOff);
    return;
  }
  if (!off && cOff) {
    printer.printOperand(cOff);
    return;
  }
  printer.getStream() << " " << 0;
}

static void printCondBr(amdgcn::AsmPrinter &printer, CBranchInstOpInterface op,
                        Block *trueDest, Block *falseDest) {
  printer.printBranchLabel(trueDest);
  if (op->getBlock()->getNextNode() == falseDest)
    return;
  printer.getStream()
      << " ; false dest is in non-adjacent block, adding s_branch\n";
  // NOTE: This is needed as MLIR doesn't allow two terminators in the same
  // block.
  printer.getStream() << "s_branch " << printer.getBranchLabel(falseDest);
}

/// Print the memory scope the LLVM assembler expects in text.
static void printScope(amdgcn::AsmPrinter &printer, AMDGCNInstOpInterface op,
                       bool sc0, bool sc1) {
  switch ((sc1 ? 2 : 0) | (sc0 ? 1 : 0)) {
  case 1:
    printer.getStream() << " scope:SCOPE_SE";
    break;
  case 2:
    printer.getStream() << " scope:SCOPE_DEV";
    break;
  case 3:
    printer.getStream() << " scope:SCOPE_SYS";
    break;
  default:
    break; // SCOPE_CU (0): elided.
  }
}

/// Print the load temporal hint the LLVM assembler expects in text.
// Note: ASTER's single NT bit only spans TH_RT (NT=0, the default, elided) and
// TH_NT (NT=1).
// TODO: add the richer hints (HT/LU/WB/compound) when needed
static void printLoadTemporality(amdgcn::AsmPrinter &printer,
                                 AMDGCNInstOpInterface op, bool nt) {
  if (nt)
    printer.getStream() << " th:TH_LOAD_NT";
}

#include "AMDGCNInstAsmPrinter.cpp.inc"
