// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- aster-cpu-translate.cpp - contrib/cpu translate driver -------------===//
//
// Standalone translate tool for the contrib/cpu AMX dialect.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMX/Registration.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::aster::amx::registerTranslateToAMXAsm();
  return mlir::failed(mlir::mlirTranslateMain(
      argc, argv, "aster contrib/cpu (AMX) translation driver"));
}
