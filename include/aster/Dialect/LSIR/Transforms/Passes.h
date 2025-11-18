//===- Passes.h - LSIR passes -----------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the LSIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_H
#define ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::aster {
namespace lsir {
#define GEN_PASS_DECL
#include "aster/Dialect/LSIR/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aster/Dialect/LSIR/Transforms/Passes.h.inc"
} // namespace lsir
} // namespace mlir::aster

#endif // ASTER_DIALECT_ASTER_TRANSFORMS_PASSES_H
