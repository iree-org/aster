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
/// Register ASTER dialects.
void initDialects(DialectRegistry &registry);

/// Register upstream MLIR dialects (e.g., arith, builtin).
void initUpstreamMLIRDialects(DialectRegistry &registry);

/// Register ASTER passes.
void registerPasses();

/// Register ASTER dialects.
ASTER_EXPORTED void asterInitDialects(MlirDialectRegistry registry);
} // namespace mlir::aster

#endif // ASTER_INIT_H
