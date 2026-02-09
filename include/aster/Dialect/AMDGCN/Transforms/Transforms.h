//===- Transforms.h - AMDGCN Transform Utilities -----------------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
#define ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H

namespace mlir {
class Operation;
class DataFlowSolver;
namespace aster {
namespace amdgcn {
/// Run register dead code elimination on the given operation using the
/// provided liveness solver. This function expects the liveness analysis to be
/// run before calling this function.
void registerDCE(Operation *op, DataFlowSolver &solver);
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
