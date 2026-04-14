// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMX/Registration.h"

#include "aster/Dialect/AMX/IR/AMXDialect.h"
#include "aster/Dialect/AMX/Target/TranslateToAsm.h"
#include "aster/Dialect/X86/IR/X86Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

void mlir::aster::amx::registerTranslateToAMXAsm() {
  static TranslateFromMLIRRegistration registration(
      "mlir-to-amx-asm", "Translate AMX dialect to x86 AMX ASM text",
      [](Operation *op, raw_ostream &os) -> LogicalResult {
        return aster::amx::translateToAsm(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<aster::x86::X86Dialect>();
        registry.insert<aster::amx::AMXDialect>();
        registry.insert<func::FuncDialect>();
      });
}
