// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- aster-cpu-opt.cpp - contrib/cpu optimizer driver -------------------===//
//
// Minimal aster-opt flavour for the contrib/cpu AMX dialect. Registers the
// `amx` dialect plus the upstream `func` dialect (needed by the fixture
// IR); no AMDGCN, no Layout, no LSIR, no test dialects or passes.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMX/IR/AMXDialect.h"
#include "aster/Dialect/X86/IR/X86Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::aster::x86::X86Dialect>();
  registry.insert<mlir::aster::amx::AMXDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "contrib/cpu (amx) optimizer driver\n", registry));
}
