//===- InstOpInterface.h - InstOp interface ---------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the aster instruction operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACE_H
#define ASTER_INTERFACES_INSTOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class OpBuilder;
namespace aster {
class InstOpInterface;
namespace detail {
/// Returns the speculatability of the instruction. If the instruction has pure
/// value semantics, this function returns Speculatable. Otherwise, it returns
/// NotSpeculatable.
Speculation::Speculatability getInstSpeculatabilityImpl(InstOpInterface op);

/// Returns the memory effects of the instruction. For each register operand
/// without value semantics this function adds read or write effects in the
/// corresponding resource.
void getInstEffectsImpl(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

/// Verify the instruction operation.
LogicalResult verifyInstImpl(InstOpInterface op);

/// Returns true if all the register operands have value semantics.
bool hasPureValueSemanticsImpl(InstOpInterface op);

/// Clones the instruction operation with new operands and results.
InstOpInterface cloneInstOpImpl(InstOpInterface op, OpBuilder &builder,
                                ValueRange outs, ValueRange ins,
                                std::optional<TypeRange> resultTypes);
struct InstAttrStorage;
} // namespace detail
} // namespace aster
} // namespace mlir

#include "aster/Interfaces/InstOpInterface.h.inc"

#endif // ASTER_INTERFACES_INSTOPINTERFACE_H
