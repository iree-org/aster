// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- mlir-air-opt.cpp - mlir-air optimizer driver -----------------------===//

#include "aster/Init.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::aster::mlir_air {
void registerAll(DialectRegistry &registry);
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Base ASTER.
  mlir::aster::registerUpstreamMLIRPasses();
  mlir::aster::initUpstreamMLIRDialects(registry);
  mlir::aster::registerUpstreamMLIRInterfaces(registry);
  mlir::aster::registerUpstreamMLIRExternalModels(registry);
  mlir::aster::initDialects(registry);
  mlir::aster::registerPasses();
  // mlir-air additions.
  mlir::aster::mlir_air::registerAll(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mlir-air optimizer driver\n", registry));
}
