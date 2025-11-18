//===- aster-opt.cpp - ASTER Optimizer Driver -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aster-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "aster/Init.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  aster::registerPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  aster::initDialects(registry);
  aster::initUpstreamMLIRDialects(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "aster modular optimizer driver\n", registry));
}
