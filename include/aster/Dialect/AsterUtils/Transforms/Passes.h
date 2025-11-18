//===- Passes.h - AsterUtils passes -----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the AsterUtils dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_H
#define ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DECL
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_H
