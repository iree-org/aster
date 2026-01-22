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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Forward declaration for test pass registration
namespace mlir::aster::test {
void registerTestDPSAliasAnalysisPass();
void registerTestLivenessAnalysisPass();
void registerTestMemoryDependenceAnalysisPass();
} // namespace mlir::aster::test

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {

  DialectRegistry registry;
  /// Upstream MLIR C++ stuff
  aster::registerUpstreamMLIRPasses();
  aster::initUpstreamMLIRDialects(registry);
  aster::registerUpstreamMLIRInterfaces(registry);
  aster::registerUpstreamMLIRExternalModels(registry);
  /// Aster C++ stuff
  aster::initDialects(registry);
  aster::registerPasses();
  aster::test::registerTestDPSAliasAnalysisPass();
  aster::test::registerTestLivenessAnalysisPass();
  aster::test::registerTestMemoryDependenceAnalysisPass();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "aster modular optimizer driver\n", registry));
}
