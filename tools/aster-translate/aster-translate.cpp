//===- aster-translate.cpp - ASTER Translate Driver -----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Init.h"
#include "aster/Target/ASM/TranslateModule.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Register AMDGCN dialect and translations.
static void registerToASMTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-asm", "Translate MLIR to AMDGCN ASM",
      [](mlir::ModuleOp topMod, raw_ostream &os) -> LogicalResult {
        if (topMod
                .walk([&](amdgcn::ModuleOp mod) {
                  return failed(target::translateModule(mod, os))
                             ? WalkResult::interrupt()
                             : WalkResult::advance();
                })
                .wasInterrupted())
          return failure();
        return success();
      },
      [](DialectRegistry &registry) {
        initDialects(registry);
        initUpstreamMLIRDialects(registry);
      });
}

int main(int argc, char **argv) {
  registerToASMTranslation();
  return failed(mlirTranslateMain(argc, argv, "AMDGCN Translation Tool"));
}
