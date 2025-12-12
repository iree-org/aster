//===- API.cpp ------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "api.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::aster::pdl {
ASTER_EXPORTED void registerDialects(MlirDialectRegistry registry) {
  DialectRegistry &dialectRegistry = *unwrap(registry);
  dialectRegistry.insert<mlir::pdl::PDLDialect, arith::ArithDialect,
                         func::FuncDialect, scf::SCFDialect, BuiltinDialect>();
}

ASTER_EXPORTED void registerPasses() {
  mlir::registerTransformsPasses();
  registerConvertPDLToPDLInterpPass();
}
} // namespace mlir::aster::pdl
