//===- Passes.h - Pass Construction and Registration ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_TRANSFORM_PASSES_H
#define AMDGCN_TRANSFORM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::aster {
namespace amdgcn {
/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

#endif // AMDGCN_TRANSFORM_PASSES_H
