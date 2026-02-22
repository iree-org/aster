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

#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Support/IntEquivalenceClasses.h"

#include <optional>

namespace mlir {
class Operation;
class DataFlowSolver;
namespace aster {
namespace amdgcn {
/// Run register dead code elimination on the given operation using the
/// provided liveness solver. This function expects the liveness analysis to be
/// run before calling this function.
void registerDCE(Operation *op, DataFlowSolver &solver);

/// Optimize the register interference graph and return equivalence classes
/// (e.g. for coalescing). Returns std::nullopt if optimization does not
/// apply or fails. The dataflow solver is expected to be loaded with the
/// reaching definitions analysis tracking only loads.
std::optional<IntEquivalenceClasses>
optimizeGraph(Operation *op, const RegisterInterferenceGraph &graph,
              DataFlowSolver &solver);
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_TRANSFORMS_TRANSFORMS_H
