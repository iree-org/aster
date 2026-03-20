//===- Transforms.cpp - Common AsterUtils transforms ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/Transforms/Transforms.h"

#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;
