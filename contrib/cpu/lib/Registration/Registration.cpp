// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/X86/Registration.h"

#include "aster/Dialect/X86/IR/X86Dialect.h"
#include "aster/Dialect/X86/Target/TranslateToAsm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

void mlir::aster::x86::registerTranslateToX86Asm() {
  static TranslateFromMLIRRegistration registration(
      "mlir-to-x86-asm", "Translate X86 dialect to x86_64 AT&T ASM text",
      [](Operation *op, raw_ostream &os) -> LogicalResult {
        return aster::x86::translateToAsm(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<aster::x86::X86Dialect>();
        registry.insert<func::FuncDialect>();
      });
}
