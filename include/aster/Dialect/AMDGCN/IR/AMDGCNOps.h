//===- AMDGCN.h - AMDGCN Dialect --------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_AMDGCN_H
#define ASTER_DIALECT_AMDGCN_AMDGCN_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h.inc"

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h.inc"

namespace mlir::aster {
namespace amdgcn {
/// Get the register kind as an integer from the given register type.
/// This call asserts if type is not an AMD register.
RegisterKind getRegisterKind(RegisterTypeInterface type);

class InstAttr;
namespace detail {
struct InstAttrStorage;
} // namespace detail
} // namespace amdgcn
} // namespace mlir::aster

#include "aster/Dialect/AMDGCN/IR/AMDGCNInstOpInterface.h.inc"

namespace mlir::aster {
namespace amdgcn {
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
} // namespace amdgcn
} // namespace mlir::aster

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h.inc"

namespace mlir::aster {
namespace amdgcn {
/// SGPR register resource.
class SGPRResource : public SideEffects::Resource::Base<SGPRResource> {
public:
  SGPRResource(SGPRType type) : type(type) {}

  static SGPRResource *get(SGPRType type);
  StringRef getName() override { return type.name; }

private:
  SGPRType type;
};

/// VGPR register resource.
class VGPRResource : public SideEffects::Resource::Base<VGPRResource> {
public:
  VGPRResource(VGPRType type) : type(type) {}

  static VGPRResource *get(VGPRType type);
  StringRef getName() override { return type.name; }

private:
  VGPRType type;
};

/// AGPR register resource.
class AGPRResource : public SideEffects::Resource::Base<AGPRResource> {
public:
  AGPRResource(AGPRType type) : type(type) {}

  static AGPRResource *get(AGPRType type);
  StringRef getName() override { return type.name; }

private:
  AGPRType type;
};

/// Global memory resource.
class GlobalMemoryResource
    : public SideEffects::Resource::Base<GlobalMemoryResource> {
public:
  GlobalMemoryResource() = default;

  static GlobalMemoryResource *get(MLIRContext *ctx);
  StringRef getName() override { return "GlobalMemory"; }
};

/// LDS memory resource (LDS - Local Data Share).
class LDSMemoryResource
    : public SideEffects::Resource::Base<LDSMemoryResource> {
public:
  LDSMemoryResource() = default;

  static LDSMemoryResource *get(MLIRContext *ctx);
  StringRef getName() override { return "LDSMemory"; }
};
} // namespace amdgcn
} // namespace mlir::aster

#define GET_OP_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h.inc"

#define AMDGCN_GEN_INST_DECLS
#include "aster/Dialect/AMDGCN/IR/AMDGCNInsts.h.inc"

#endif // ASTER_DIALECT_AMDGCN_AMDGCN_H
