//===- API.h --------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_PDL_API_H
#define ASTER_PDL_API_H

#include "aster/Support/API.h"
#include "mlir-c/IR.h"

namespace mlir::aster::pdl {
/// Register all PDL related dialects in the given registry.
ASTER_EXPORTED void registerDialects(MlirDialectRegistry registry);
/// Register all PDL related passes.
ASTER_EXPORTED void registerPasses();
} // namespace mlir::aster::pdl

#endif // ASTER_PDL_API_H
