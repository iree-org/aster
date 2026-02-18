//===- Passes.h - Test Pass Construction and Registration -----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::aster {
namespace test {
/// Generate the code for declaring the passes.
#define GEN_PASS_DECL
#include "Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"
} // namespace test

void registerTestPasses();
} // namespace mlir::aster
