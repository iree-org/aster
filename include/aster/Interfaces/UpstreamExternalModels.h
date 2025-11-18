//===- GPUFuncInterface.td --------------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_UPSTREAMEXTERNALMODELS_H
#define ASTER_INTERFACES_UPSTREAMEXTERNALMODELS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::aster {
/// Register external model implementations for various upstream MLIR
/// interfaces.
void registerUpstreamExternalModels(DialectRegistry &registry);
} // namespace mlir::aster

#endif // ASTER_INTERFACES_UPSTREAMEXTERNALMODELS_H
