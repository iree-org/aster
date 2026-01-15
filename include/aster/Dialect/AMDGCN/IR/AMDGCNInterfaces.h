//===- AMDGCNInterfaces.h - AMDGCN Interfaces -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNINTERFACES_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNINTERFACES_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNInstOpInterface.h.inc"

namespace mlir::aster::amdgcn {
/// Returns the speculatability of the operation.
Speculation::Speculatability getInstSpeculatability(InstOpInterface op);

/// Returns the memory effects of the operation.
void getInstEffects(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

/// Trait to provide utility methods for instruction operations.
template <typename ConcreteType>
struct InstOpTrait : public OpTrait::TraitBase<ConcreteType, InstOpTrait> {
  /// Get the number of output operands.
  static size_t getNumOuts(ConcreteType op, size_t numOuts) {
    size_t c = 0;
    for (size_t i = 0; i < numOuts; ++i) {
      auto [start, size] = op.getODSOperandIndexAndLength(i);
      c += size;
    }
    return c;
  }
  /// Get the number of input operands.
  static size_t getNumIns(ConcreteType op, size_t numOuts, size_t numIns) {
    size_t c = 0;
    for (size_t i = 0; i < numIns; ++i) {
      auto [start, size] = op.getODSOperandIndexAndLength(i + numOuts);
      c += size;
    }
    return c;
  }
};

/// Global memory resource.
class GlobalMemoryResource
    : public SideEffects::Resource::Base<GlobalMemoryResource> {
public:
  StringRef getName() override { return "amdgcn.global_memory"; }
};

/// LDS memory resource (LDS - Local Data Share).
class LDSMemoryResource
    : public SideEffects::Resource::Base<LDSMemoryResource> {
public:
  StringRef getName() override { return "amdgcn.lds_memory"; }
};
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNINTERFACES_H
