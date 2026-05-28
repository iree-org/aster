//===- Scheduler.h - AMDGCN scheduler external model registration ---------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULER_H
#define ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULER_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::aster::amdgcn {
void registerAMDGCNSchedulerExternalModels(DialectRegistry &registry);
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_SCHEDULER_SCHEDULER_H
