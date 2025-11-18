//===- AMDGCNVerifiers.h - AMDGCN verifier attributes ---------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN verifier implementation.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H
#define ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H

#include "aster/Interfaces/VerifierAttr.h"

namespace mlir::aster::amdgcn {
/// Check that an operation is inside an AMDGCN kernel.
LogicalResult verifyIsInKernelImpl(Operation *op, const VerifierState &state);
/// Check that an operation is inside an AMDGCN module.
LogicalResult verifyIsInModuleImpl(Operation *op, const VerifierState &state);
/// Check that an operation can be allocated by the register allocator.
LogicalResult verifyIsAllocatableOpImpl(Operation *op,
                                        const VerifierState &state);
/// Check that an operation can be translated to assembly.
LogicalResult verifyIsAllocatedOpImpl(Operation *op,
                                      const VerifierState &state);
/// Check constants used in AMDGCN operations.
LogicalResult verifyConstantsImpl(Operation *op, const VerifierState &state);
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H
