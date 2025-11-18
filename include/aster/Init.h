//===- Init.h -------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INIT_H
#define ASTER_INIT_H

#include "aster/Support/API.h"
#include "mlir-c/IR.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::aster {
///
/// Upstream MLIR C++ stuff
///
void initUpstreamMLIRDialects(DialectRegistry &registry);
void registerUpstreamMLIRInterfaces(DialectRegistry &registry);
void registerUpstreamMLIRExternalModels(DialectRegistry &registry);
void registerUpstreamMLIRPasses();

///
/// Upstream MLIR CAPI stuff
///
ASTER_EXPORTED void
asterRegisterUpstreamMLIRDialects(MlirDialectRegistry registry);
ASTER_EXPORTED void
asterRegisterUpstreamMLIRInterfaces(MlirDialectRegistry registry);
ASTER_EXPORTED void
asterRegisterUpstreamMLIRInterfaces(MlirDialectRegistry registry);
ASTER_EXPORTED void
asterRegisterUpstreamMLIRExternalModels(MlirDialectRegistry registry);

///
/// Aster C++ stuff
///
void initDialects(DialectRegistry &registry);
void registerPasses();

///
/// Aster CAPI stuff
///
ASTER_EXPORTED void asterInitDialects(MlirDialectRegistry registry);
} // namespace mlir::aster

#endif // ASTER_INIT_H
