//===- TranslateModule.h - Export ASM ---------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines translation utilities for the export ASM target.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TARGET_ASM_TRANSLATEMODULE_H
#define ASTER_TARGET_ASM_TRANSLATEMODULE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::aster {
class TargetAttrInterface;
namespace amdgcn {
class AsmPrinter;
class ModuleOp;
namespace target {

LogicalResult printISAInstruction(amdgcn::AsmPrinter &printer,
                                  TargetAttrInterface tgt, Operation *op);

// Structure to hold kernel argument metadata
struct KernelArg {
  unsigned size;
  unsigned offset;
  std::string valueKind;
  std::string addressSpace;
  std::string actualAccess;
};

// Translates the given AMDGCN module to AMDGPU assembly and writes it to the
// provided output stream.
LogicalResult translateModule(mlir::aster::amdgcn::ModuleOp module,
                              llvm::raw_ostream &os, bool debugPrint = false);
} // namespace target
} // namespace amdgcn
} // namespace mlir::aster

#endif // ASTER_TARGET_ASM_TRANSLATEMODULE_H
