//===- Passes.cpp - Test Pass Construction and Registration ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

namespace mlir::aster {
void registerTestPasses() { test::registerPasses(); }
} // namespace mlir::aster
