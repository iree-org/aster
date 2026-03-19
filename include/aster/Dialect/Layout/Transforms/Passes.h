//===- Passes.h - Layout passes -------------------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_H
#define ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::aster {
namespace layout {
#define GEN_PASS_DECL
#include "aster/Dialect/Layout/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "aster/Dialect/Layout/Transforms/Passes.h.inc"
} // namespace layout
} // namespace mlir::aster

#endif // ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_H
