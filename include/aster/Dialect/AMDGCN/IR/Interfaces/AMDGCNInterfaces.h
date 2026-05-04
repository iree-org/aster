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

#ifndef ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINTERFACES_H
#define ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINTERFACES_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::aster::amdgcn {
/// Constant memory resource (CONST - Constant Memory).
class ConstantMemoryResource
    : public SideEffects::Resource::Base<ConstantMemoryResource> {
public:
  StringRef getName() const override { return "amdgcn.constant_memory"; }
};

/// Global memory resource.
class GlobalMemoryResource
    : public SideEffects::Resource::Base<GlobalMemoryResource> {
public:
  StringRef getName() const override { return "amdgcn.global_memory"; }
};

/// LDS memory resource (LDS - Local Data Share).
class LDSMemoryResource
    : public SideEffects::Resource::Base<LDSMemoryResource> {
public:
  StringRef getName() const override { return "amdgcn.lds_memory"; }
};

/// Memory unit issuer resource.
class MemoryUnit : public SideEffects::Resource::Base<MemoryUnit> {
public:
  StringRef getName() const override { return "amdgcn.memory_unit"; }
};

/// Scalar Unit (SALU) resource.
class SALUResource : public SideEffects::Resource::Base<SALUResource> {
public:
  StringRef getName() const override { return "amdgcn.salu_resource"; }
};

/// Vector Unit (VALU) resource.
class VALUResource : public SideEffects::Resource::Base<VALUResource> {
public:
  StringRef getName() const override { return "amdgcn.valu_resource"; }
};
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINTERFACES_H
