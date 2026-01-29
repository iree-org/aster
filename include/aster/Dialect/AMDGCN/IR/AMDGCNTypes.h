//===- AMDGCNTypes.h - AMDGCN Types -----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/DependentOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Interfaces/ResourceInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNTypeInterfaces.h.inc"

namespace mlir::aster::amdgcn {
/// SREG register resource.
class SREGResource : public SideEffects::Resource::Base<SREGResource> {
public:
  StringRef getName() override { return "amdgcn.special_register"; }
};
} // namespace mlir::aster::amdgcn

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h.inc"

namespace mlir::aster::amdgcn {
/// GGPR register resource.
class GGPRResource : public SideEffects::Resource::Base<GGPRResource> {
public:
  StringRef getName() override { return GGPRType::name; }
};

/// SGPR register resource.
class SGPRResource : public SideEffects::Resource::Base<SGPRResource> {
public:
  StringRef getName() override { return SGPRType::name; }
};

/// VGPR register resource.
class VGPRResource : public SideEffects::Resource::Base<VGPRResource> {
public:
  StringRef getName() override { return VGPRType::name; }
};

/// AGPR register resource.
class AGPRResource : public SideEffects::Resource::Base<AGPRResource> {
public:
  StringRef getName() override { return AGPRType::name; }
};

/// Get the register kind as an integer from the given register type.
/// This call asserts if type is not an AMD register.
RegisterKind getRegisterKind(AMDGCNRegisterTypeInterface type);

/// Compare two AMDGCN register types.
/// Returns true if `lhs` is less than `rhs`.
bool compareLessAMDGCNRegisterTypes(AMDGCNRegisterTypeInterface lhs,
                                    AMDGCNRegisterTypeInterface rhs);
/// Check if the given type is a register range and has the given size.
bool hasSize(Type type, ArrayRef<int32_t> size);
} // namespace mlir::aster::amdgcn

namespace mlir::aster {
/// Comparison operator for resource types based on their partial order.
/// NOTE: This operator makes a hard fail with `cast` if the types are not
/// AMDGCN register types. It's also a layering violation, as this operator
/// depends on the AMDGCN dialect. However, this is needed to use ResourceType
/// in generic data structures that require ordering (e.g., DenseMap).
inline bool operator<(ResourceTypeInterface lhs, ResourceTypeInterface rhs) {
  // Make a hard fail with `cast` if the types are not AMDGCN register types.
  return compareLessAMDGCNRegisterTypes(
      cast<amdgcn::AMDGCNRegisterTypeInterface>(lhs),
      cast<amdgcn::AMDGCNRegisterTypeInterface>(rhs));
}
} // namespace mlir::aster

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNTYPES_H
